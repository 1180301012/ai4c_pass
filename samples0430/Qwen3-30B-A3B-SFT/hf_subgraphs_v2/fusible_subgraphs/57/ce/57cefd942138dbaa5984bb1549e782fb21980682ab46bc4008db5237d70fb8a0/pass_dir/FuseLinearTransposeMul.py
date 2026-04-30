import torch
import triton
import triton.language as tl


# ── Pattern ──────────────────────────────────────────────────────────────────
def pattern(in_0, in_1, in_2, in_3):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3  = linear.transpose(-1, -2)
    tmp_4  = in_3 * tmp_3
    return tmp_4


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


# ── Triton kernel ─────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        # Best tile sizes for all batch sizes — autotuner picks optimal per (B,M,N,K)
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 32}, num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 32}, num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4,  num_stages=3),
    ],
    key=['B', 'M', 'N', 'K'],
)
@triton.jit
def _fused_linear_transpose_mul_kernel(
    # pointers
    in2_ptr,   # [B, M, K]  (linear input)
    w_ptr,     # [N, K]     (weight)
    bias_ptr,  # [N]        (bias)
    in3_ptr,   # [B, N, M]  (gate)
    out_ptr,   # [B, N, M]  (output)
    # dimensions
    B, M, N, K,
    # strides for in2  [B, M, K]  →  stride_b, stride_m, 1
    stride_i2b, stride_i2m,
    # strides for w    [N, K]     →  stride_n,  1
    stride_wn,
    # strides for in3  [B, N, M]  →  stride_b, stride_n,  1
    stride_i3b, stride_i3n,
    # strides for out  [B, N, M]  →  stride_b, stride_n,  1
    stride_ob,  stride_on,
    # tile sizes (set by autotuner)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # ── program ids ────────────────────────────────────────────────────────────
    pid_m = tl.program_id(0)   # tile index along M (rows)
    pid_n = tl.program_id(1)   # tile index along N (cols)
    pid_b = tl.program_id(2)   # batch index

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # ── GEMM accumulator ──────────────────────────────────────────────────────
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    base_in2 = in2_ptr + pid_b * stride_i2b
    base_w   = w_ptr
    base_in3 = in3_ptr + pid_b * stride_i3b
    base_out = out_ptr + pid_b * stride_ob

    # ── K loop with masking for partial last block ─────────────────────────────
    for k_start in range(0, K, BLOCK_K):
        cur_k  = k_start + offs_k
        k_mask = cur_k < K

        a = tl.load(
            base_in2 + offs_m[:, None] * stride_i2m + cur_k[None, :],
            mask=(offs_m[:, None] < M) & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        b = tl.load(
            base_w + offs_n[:, None] * stride_wn + cur_k[None, :],
            mask=(offs_n[:, None] < N) & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        acc = tl.dot(a, tl.trans(b), acc, out_dtype=tl.float32)

    # ── Bias addition ─────────────────────────────────────────────────────────
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc  = acc + bias[None, :]

    # ── Fused element-wise multiply with in3  [B, N, M] ──────────────────────
    out_mask  = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    m_for_in3 = offs_m[:, None]
    n_for_in3 = offs_n[None, :]

    in3    = tl.load(base_in3 + m_for_in3 + n_for_in3 * stride_i3n, mask=out_mask, other=0.0).to(tl.float32)
    result = acc * in3

    # ── Store  [B, N, M] ──────────────────────────────────────────────────────
    tl.store(base_out + m_for_in3 + n_for_in3 * stride_on, result, mask=out_mask)


# ── Wrapper (must be @torch.fx.wrap) ─────────────────────────────────────────
@torch.fx.wrap
def fused_linear_transpose_mul(in_0, in_1, in_2, in_3):
    """
    in_0 : bias   [N]
    in_1 : weight [N, K]
    in_2 : input  [B, M, K]
    in_3 : gate   [B, N, M]
    returns       [B, N, M]
    """
    B, M, K = in_2.shape
    N       = in_1.shape[0]

    out = torch.empty((B, N, M), dtype=in_2.dtype, device=in_2.device)

    def grid(meta):
        return (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(N, meta['BLOCK_N']),
            B,
        )

    _fused_linear_transpose_mul_kernel[grid](
        in_2, in_1, in_0, in_3, out,
        B, M, N, K,
        in_2.stride(0), in_2.stride(1),
        in_1.stride(0),
        in_3.stride(0), in_3.stride(1),
        out.stride(0),  out.stride(1),
    )
    return out


# ── Replacement factory ───────────────────────────────────────────────────────
def replacement_func():
    return fused_linear_transpose_mul