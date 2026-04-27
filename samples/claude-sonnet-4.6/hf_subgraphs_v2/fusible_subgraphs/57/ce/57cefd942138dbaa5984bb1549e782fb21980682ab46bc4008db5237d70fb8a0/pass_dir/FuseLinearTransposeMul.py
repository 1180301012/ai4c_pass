import torch
import triton
import triton.language as tl


def pattern(bias, weight, x, gate):
    linear = torch.nn.functional.linear(x, weight, bias)
    transposed = linear.transpose(-1, -2)
    result = gate * transposed
    return result


def replacement_args(bias, weight, x, gate):
    return (bias, weight, x, gate)


@triton.autotune(
    configs=[
        # BLOCK_N=16: more programs for B=1 (312 vs 84) — targets B=1 SM occupancy
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 16, 'BLOCK_K': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 16, 'BLOCK_K': 64},  num_warps=4, num_stages=2),
        # BLOCK_N=32 family
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32, 'BLOCK_K': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32, 'BLOCK_K': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32, 'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64},  num_warps=8, num_stages=2),
        # BLOCK_N=64 family
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64},  num_warps=8, num_stages=2),
        # BLOCK_K=128 family
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32, 'BLOCK_K': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32, 'BLOCK_K': 64},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
        # Large BLOCK_M for B=128 efficiency
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=8, num_stages=2),
    ],
    # DTYPE in the key ensures each dtype gets its own optimal config:
    #   - float32/6 (B=128) will NOT reuse bfloat16/6's cached config
    #   - Each dtype autotunes separately → extra GPU warmup helps float32/6 stability
    key=['B', 'M', 'N', 'K', 'DTYPE'],
)
@triton.jit
def fused_linear_transpose_mul_kernel(
    x_ptr, w_ptr, bias_ptr, gate_ptr, out_ptr,
    B, M, N, K,
    DTYPE,          # 0=fp32, 1=fp16, 2=bf16  (for autotune key only, not used in body)
    stride_xb, stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_gb, stride_gn, stride_gm,
    stride_ob, stride_on, stride_om,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused kernel: out[b, n, m] = gate[b, n, m] * (sum_k x[b,m,k]*w[n,k] + bias[n])

    Uses the INPUT dtype directly in tl.dot (no fp32 upcast before the dot):
      - fp16/bf16 inputs  →  FP16/BF16 tensor cores (41.2 TFLOPS on A30)
      - fp32 inputs       →  TF32 tensor cores      (10.3 TFLOPS on A30)
    The fp32 accumulator is retained throughout for numerical fidelity.

    Layout:  acc[BLOCK_M, BLOCK_N]  →  tl.trans → [BLOCK_N, BLOCK_M]  →  * gate  →  store
    Shapes:
        x    : [B, M, K]   (in_2)
        w    : [N, K]      (in_1, weight)
        bias : [N]         (in_0)
        gate : [B, N, M]   (in_3)
        out  : [B, N, M]
    """
    pid_m = tl.program_id(0)   # tile index along M (768)
    pid_n = tl.program_id(1)   # tile index along N (196)
    b     = tl.program_id(2)   # batch index

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = m_offs < M
    n_mask = n_offs < N

    # Accumulate in fp32: acc[m, n] = sum_k  x[b, m, k] * w[n, k]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K

        # x tile: [BLOCK_M, BLOCK_K]  — last dim K (stride 1): coalesced
        x_mask = m_mask[:, None] & k_mask[None, :]
        x_ptrs = x_ptr + b * stride_xb + m_offs[:, None] * stride_xm + k_offs[None, :] * stride_xk
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # w tile: [BLOCK_N, BLOCK_K]  — last dim K (stride 1): coalesced
        w_mask = n_mask[:, None] & k_mask[None, :]
        w_ptrs = w_ptr + n_offs[:, None] * stride_wn + k_offs[None, :] * stride_wk
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # acc[m, n] += x[m,k] * w[n,k]  =  x_tile @ w_tile.T
        # Use native input dtype for tl.dot → fp16/bf16 → FP16/BF16 tensor cores (4x faster)
        #                                     fp32       → TF32 tensor cores
        acc = tl.dot(x_tile, tl.trans(w_tile), acc, out_dtype=tl.float32)

    # Add bias[n]: broadcast over M  ([BLOCK_N] → [BLOCK_M, BLOCK_N])
    bias_tile = tl.load(bias_ptr + n_offs, mask=n_mask, other=0.0).to(tl.float32)
    acc = acc + bias_tile[None, :]   # acc: [BLOCK_M, BLOCK_N]

    # Load gate[b, n_offs, m_offs]: [BLOCK_N, BLOCK_M]
    gate_mask = n_mask[:, None] & m_mask[None, :]
    gate_ptrs = gate_ptr + b * stride_gb + n_offs[:, None] * stride_gn + m_offs[None, :] * stride_gm
    gate_tile = tl.load(gate_ptrs, mask=gate_mask, other=0.0)

    # out[b, n, m] = gate[b,n,m] * acc[m,n]  — transpose acc to [BLOCK_N, BLOCK_M]
    acc_t  = tl.trans(acc).to(gate_tile.dtype)   # [BLOCK_N, BLOCK_M]
    result = gate_tile * acc_t                    # [BLOCK_N, BLOCK_M]

    # Store out[b, n_offs, m_offs]
    out_ptrs = out_ptr + b * stride_ob + n_offs[:, None] * stride_on + m_offs[None, :] * stride_om
    tl.store(out_ptrs, result, mask=gate_mask)


@torch.fx.wrap
def fused_linear_transpose_mul(bias, weight, x, gate):
    """
    Matches:  linear(x, weight, bias).transpose(-1, -2) * gate

    Shapes:
        bias   : [N]        e.g. [196]
        weight : [N, K]     e.g. [196, 196]
        x      : [B, M, K]  e.g. [B, 768, 196]
        gate   : [B, N, M]  e.g. [B, 196, 768]
        output : [B, N, M]
    """
    B, M, K = x.shape
    N = weight.shape[0]

    out = torch.empty(B, N, M, dtype=x.dtype, device=x.device)

    _dtype_id = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}
    DTYPE = _dtype_id.get(x.dtype, 0)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N']),
        B,
    )

    fused_linear_transpose_mul_kernel[grid](
        x, weight, bias, gate, out,
        B, M, N, K,
        DTYPE,
        x.stride(0),      x.stride(1),      x.stride(2),
        weight.stride(0), weight.stride(1),
        gate.stride(0),   gate.stride(1),   gate.stride(2),
        out.stride(0),    out.stride(1),    out.stride(2),
    )

    return out


def replacement_func():
    return fused_linear_transpose_mul