import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: scale * in_0  →  softmax(dim=-1)
#
#   in_0 : [B, M, K]  e.g. [B, 8192, 19]
#   out  : [B, M, K]  (softmax output in the same shape)
#
# Fuses the 0.0625 scale and the softmax into a single memory pass.
# K=19 fits entirely in registers so no shared memory is needed.
# ---------------------------------------------------------------------------

def pattern(in_0):
    tmp_0 = 0.0625 * in_0
    tmp_1 = torch.nn.functional.softmax(tmp_0, dim=-1)
    return tmp_1


def replacement_args(in_0):
    return (in_0,)


# ---------------------------------------------------------------------------
# Triton kernel: fused scale + row-softmax
#   Grid: (B * ceil(M / BLOCK_M),)  — each CTA handles BLOCK_M rows
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_M': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_M': 512, 'BLOCK_K': 32}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_M': 64,  'BLOCK_K': 32}, num_warps=8, num_stages=1),
    ],
    key=['B', 'M', 'K'],
)
@triton.jit
def _scale_softmax_kernel(
    in_ptr,   # [B, M, K]
    out_ptr,  # [B, M, K]
    B, M, K,
    stride_b, stride_m, stride_k,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # pid_bm encodes (batch, m_tile)
    pid_bm = tl.program_id(0)
    num_m_tiles = tl.cdiv(M, BLOCK_M)
    pid_b  = pid_bm // num_m_tiles
    pid_m  = pid_bm %  num_m_tiles

    m_start = pid_m * BLOCK_M
    m_range = m_start + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    m_mask  = m_range < M

    k_range = tl.arange(0, BLOCK_K)             # [BLOCK_K]
    k_mask  = k_range < K

    # Load [BLOCK_M, BLOCK_K] and convert to fp32
    base    = pid_b * stride_b
    offsets = base + m_range[:, None] * stride_m + k_range[None, :] * stride_k
    mask    = m_mask[:, None] & k_mask[None, :]

    a = tl.load(in_ptr + offsets, mask=mask, other=float('-inf')).to(tl.float32)

    # Scale
    a = a * 0.0625

    # Zero-out padding positions in softmax denominator
    a = tl.where(k_mask[None, :], a, float('-inf'))

    # Numerically stable row-softmax over K
    a_max = tl.max(a, axis=1, keep_dims=True)          # [BLOCK_M, 1]
    a_exp = tl.exp(a - a_max)                           # [BLOCK_M, BLOCK_K]
    a_exp = tl.where(k_mask[None, :], a_exp, 0.0)
    a_sum = tl.sum(a_exp, axis=1, keep_dims=True)       # [BLOCK_M, 1]
    result = a_exp / a_sum                              # [BLOCK_M, BLOCK_K]

    # Cast back and store
    if IS_FP16:
        result = result.to(tl.float16)
    elif IS_BF16:
        result = result.to(tl.bfloat16)

    tl.store(out_ptr + offsets, result, mask=mask)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_scale_softmax(in_0):
    B, M, K = in_0.shape

    out = torch.empty_like(in_0)

    IS_FP16 = (in_0.dtype == torch.float16)
    IS_BF16 = (in_0.dtype == torch.bfloat16)

    # BLOCK_K: smallest power-of-2 >= K (K=19 → 32)
    BLOCK_K = max(triton.next_power_of_2(K), 16)

    def grid(meta):
        return (B * triton.cdiv(M, meta['BLOCK_M']),)

    _scale_softmax_kernel[grid](
        in_0, out,
        B, M, K,
        in_0.stride(0), in_0.stride(1), in_0.stride(2),
        IS_FP16=IS_FP16,
        IS_BF16=IS_BF16,
        BLOCK_K=BLOCK_K,
    )

    return out


# ---------------------------------------------------------------------------
# Required by the AI4C framework
# ---------------------------------------------------------------------------

def replacement_func():
    return fused_scale_softmax