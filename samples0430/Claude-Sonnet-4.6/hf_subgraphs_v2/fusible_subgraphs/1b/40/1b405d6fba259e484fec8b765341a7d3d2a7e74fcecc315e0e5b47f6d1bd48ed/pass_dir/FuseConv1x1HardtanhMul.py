import torch
import triton
import triton.language as tl


# ──────────────────────────────────────────────────────────────────────────────
# Pattern & replacement_args
# ──────────────────────────────────────────────────────────────────────────────

def pattern(in_0, in_1, in_2, in_3):
    """
    Matches:
      conv2d = torch.conv2d(in_2, in_1, in_0, (1,1), (0,0), (1,1), 1)
      tmp_3  = hardtanh(in_3, 0.0, 6.0, False)
      tmp_4  = tmp_3 * conv2d
    """
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3  = torch.nn.functional.hardtanh(in_3, 0.0, 6.0, False)
    tmp_4  = tmp_3 * conv2d
    return tmp_4


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)



# ──────────────────────────────────────────────────────────────────────────────
# Triton kernel
#
# Fused 1×1-conv (as GEMM) + hardtanh(in_3, 0, 6) + elementwise multiply.
#
# Memory layout for in_2 (NCHW):  element [n, c, hw] = n*C_in*HW + c*HW + hw
# Memory layout for in_3 (NCHW):  element [n, c, hw] = n*C_out*HW + c*HW + hw
# weight is [C_out, C_in] (after reshaping the 1×1 kernel)
# ──────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32,  'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
    ],
    key=['M', 'C_in', 'C_out', 'HW'],
)
@triton.jit
def _fused_conv1x1_ht_mul(
    in2_ptr,    # [N, C_in,  H, W]  CUDA
    wt_ptr,     # [C_out, C_in]     CUDA
    bias_ptr,   # [C_out]           CUDA
    in3_ptr,    # [N, C_out, H, W]  CUDA
    out_ptr,    # [N, C_out, H, W]  CUDA  (output)
    M,          # = N * H * W  (number of "rows" for the GEMM)
    C_in,       # = 24
    C_out,      # = 96
    HW,         # = H * W
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    OUT_DTYPE:  tl.constexpr,   # tl.float16 / tl.bfloat16 / tl.float32
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_offs = m_start + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    n_offs = n_start + tl.arange(0, BLOCK_N)   # [BLOCK_N]

    # Decompose flat m index into (batch_idx, hw_idx) for NCHW addressing.
    hw_idx  = m_offs % HW           # [BLOCK_M]
    bat_idx = m_offs // HW          # [BLOCK_M]

    # ── Accumulator in fp32 for numerical stability ──────────────────────────
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ── K-loop over input channels ──────────────────────────────────────────
    for k_start in range(0, C_in, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)   # [BLOCK_K]

        # Load A tile from in_2 (NCHW): in2[bat, c, hw] = bat*C_in*HW + c*HW + hw
        a_offs = (bat_idx[:, None] * (C_in * HW)
                  + k_offs[None, :] * HW
                  + hw_idx[:, None])
        a_mask = (m_offs[:, None] < M) & (k_offs[None, :] < C_in)
        a = tl.load(in2_ptr + a_offs, mask=a_mask, other=0.0)

        # Load B tile from weight [C_out, C_in]: weight[c_out, c_in] = c_out*C_in + c_in
        b_offs = n_offs[None, :] * C_in + k_offs[:, None]
        b_mask = (n_offs[None, :] < C_out) & (k_offs[:, None] < C_in)
        b = tl.load(wt_ptr + b_offs, mask=b_mask, other=0.0)

        # Accumulate using native dtype with fp32 output.
        # For fp16/bf16 this exploits fp16 tensor cores (much faster);
        # for fp32 this uses TF32 tensor cores via allow_tf32.
        acc += tl.dot(a, b, out_dtype=tl.float32)

    # ── Add bias ────────────────────────────────────────────────────────────
    bias = tl.load(bias_ptr + n_offs,
                   mask=(n_offs < C_out), other=0.0).to(tl.float32)
    acc += bias[None, :]

    # ── Load in_3 (NCHW), apply hardtanh(0, 6), multiply ───────────────────
    in3_offs = (bat_idx[:, None] * (C_out * HW)
                + n_offs[None, :] * HW
                + hw_idx[:, None])
    in3_mask = (m_offs[:, None] < M) & (n_offs[None, :] < C_out)
    in3 = tl.load(in3_ptr + in3_offs, mask=in3_mask, other=0.0).to(tl.float32)

    # hardtanh(in3, 0.0, 6.0) == clamp(in3, 0, 6)
    in3_clamped = tl.minimum(tl.maximum(in3, 0.0), 6.0)

    result = in3_clamped * acc

    # ── Store result ─────────────────────────────────────────────────────────
    tl.store(out_ptr + in3_offs,
             result.to(OUT_DTYPE),
             mask=in3_mask)


# ──────────────────────────────────────────────────────────────────────────────
# Python wrapper
# ──────────────────────────────────────────────────────────────────────────────

@torch.fx.wrap
def conv1x1_hardtanh_mul(in_0, in_1, in_2, in_3):
    """
    in_0 : bias   [C_out]            (may be on CPU)
    in_1 : weight [C_out, C_in,1,1]  (may be on CPU)
    in_2 : input  [N, C_in, H, W]    CUDA
    in_3 : other  [N, C_out, H, W]   CUDA
    """
    device = in_2.device
    dtype  = in_2.dtype

    # Move weights to the GPU device.  torch.as_tensor is whitelisted.
    # weight shape [C_out, C_in, 1, 1] has the same flat memory layout as
    # [C_out, C_in] (strides are [..., 1, 1]), so no reshape is needed.
    bias   = torch.as_tensor(in_0, device=device, dtype=dtype)
    weight = torch.as_tensor(in_1, device=device, dtype=dtype)

    N, C_in, H, W = in_2.shape
    C_out = weight.shape[0]
    HW    = H * W
    M     = N * HW

    out = torch.empty_like(in_3)

    # Map torch dtype → triton constexpr dtype
    if dtype == torch.float16:
        out_dtype = tl.float16
    elif dtype == torch.bfloat16:
        out_dtype = tl.bfloat16
    else:
        out_dtype = tl.float32

    grid = lambda meta: (
        triton.cdiv(M,    meta['BLOCK_M']),
        triton.cdiv(C_out, meta['BLOCK_N']),
    )

    _fused_conv1x1_ht_mul[grid](
        in_2, weight, bias, in_3, out,
        M, C_in, C_out, HW,
        OUT_DTYPE=out_dtype,
    )

    return out


def replacement_func():
    return conv1x1_hardtanh_mul