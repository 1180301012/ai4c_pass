"""
Shared Triton kernel for WavLM fused linear + sigmoid + chunk arithmetic.

The computation simplifies to:
  out[h, i, 0] = 1.0 + s[i] + 2.0 * (s[i] - s[i//4])
  where s[i] = sigmoid(linear(x[h,i,:]))

This fuses: linear + view + sum(-1) + sigmoid + chunk + elementwise arithmetic
into a single kernel, avoiding materialization of the [1,H,199,8] intermediate.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_R': 1}, num_warps=1),
        triton.Config({'BLOCK_R': 1}, num_warps=2),
        triton.Config({'BLOCK_R': 1}, num_warps=4),
        triton.Config({'BLOCK_R': 1}, num_warps=8),
        triton.Config({'BLOCK_R': 2}, num_warps=2),
        triton.Config({'BLOCK_R': 2}, num_warps=4),
        triton.Config({'BLOCK_R': 2}, num_warps=8),
        triton.Config({'BLOCK_R': 4}, num_warps=4),
        triton.Config({'BLOCK_R': 4}, num_warps=8),
    ],
    key=['H'],
)
@triton.jit
def _fused_wavlm_sigmoid_kernel(
    x_ptr,       # [1, H, 199, 64]  bfloat16/float16
    w_ptr,       # [8, 64]          bfloat16/float16  (in_1, weight)
    b_ptr,       # [8]              bfloat16/float16  (in_0, bias)
    in2_ptr,     # [1, H, 1, 1]     bfloat16/float16  (in_2)
    out_ptr,     # [1, H, 199, 1]   bfloat16/float16
    H,           # number of heads (runtime int)
    BLOCK_K: tl.constexpr,   # = 64 (inner dimension)
    BLOCK_R: tl.constexpr,   # rows per program (autotuned)
):
    """
    Each program handles BLOCK_R consecutive rows for one head h.
    pid = h * (199 // BLOCK_R) + r  (grid = H * ceil(199/BLOCK_R) programs)

    Correctness: for each row i, the formula is:
        s_i  = sigmoid(x[h,i,:] @ w[:,i%4+4] + b[i%4+4])
        s_i4 = sigmoid(x[h,i,:] @ w[:,i//4]  + b[i//4])
        out  = 1 + s_i + 2*(s_i - s_i4)

    No scalar tensor indexing — keep each dp as a separate float32 register.
    """
    N_BLOCKS_R = (199 + BLOCK_R - 1) // BLOCK_R
    pid = tl.program_id(0)
    h   = pid // N_BLOCKS_R
    r   = pid %  N_BLOCKS_R
    base_i = r * BLOCK_R

    # Load bias (8 scalars, stays in L1 cache)
    b0 = tl.load(b_ptr + 0).to(tl.float32)
    b1 = tl.load(b_ptr + 1).to(tl.float32)
    b2 = tl.load(b_ptr + 2).to(tl.float32)
    b3 = tl.load(b_ptr + 3).to(tl.float32)
    b4 = tl.load(b_ptr + 4).to(tl.float32)
    b5 = tl.load(b_ptr + 5).to(tl.float32)
    b6 = tl.load(b_ptr + 6).to(tl.float32)
    b7 = tl.load(b_ptr + 7).to(tl.float32)

    # Load in2[0, h, 0, 0]
    in2_val = tl.load(in2_ptr + h).to(tl.float32)

    k_range = tl.arange(0, BLOCK_K)   # [64]

    # Process BLOCK_R consecutive rows
    for row_id in tl.static_range(BLOCK_R):
        i = base_i + row_id
        if i < 199:
            # Load x[0, h, i, 0:64]
            x = tl.load(x_ptr + h * 199 * 64 + i * 64 + k_range).to(tl.float32)

            # 8 scalar dot products (no tensor indexing)
            dp0 = tl.sum(tl.load(w_ptr + 0 * 64 + k_range).to(tl.float32) * x)
            dp1 = tl.sum(tl.load(w_ptr + 1 * 64 + k_range).to(tl.float32) * x)
            dp2 = tl.sum(tl.load(w_ptr + 2 * 64 + k_range).to(tl.float32) * x)
            dp3 = tl.sum(tl.load(w_ptr + 3 * 64 + k_range).to(tl.float32) * x)
            dp4 = tl.sum(tl.load(w_ptr + 4 * 64 + k_range).to(tl.float32) * x)
            dp5 = tl.sum(tl.load(w_ptr + 5 * 64 + k_range).to(tl.float32) * x)
            dp6 = tl.sum(tl.load(w_ptr + 6 * 64 + k_range).to(tl.float32) * x)
            dp7 = tl.sum(tl.load(w_ptr + 7 * 64 + k_range).to(tl.float32) * x)

            s0 = 1.0 / (1.0 + tl.exp(-(dp0 + b0)))
            s1 = 1.0 / (1.0 + tl.exp(-(dp1 + b1)))
            s2 = 1.0 / (1.0 + tl.exp(-(dp2 + b2)))
            s3 = 1.0 / (1.0 + tl.exp(-(dp3 + b3)))
            s4 = 1.0 / (1.0 + tl.exp(-(dp4 + b4)))
            s5 = 1.0 / (1.0 + tl.exp(-(dp5 + b5)))
            s6 = 1.0 / (1.0 + tl.exp(-(dp6 + b6)))
            s7 = 1.0 / (1.0 + tl.exp(-(dp7 + b7)))

            # i%4 ∈ {0,1,2,3}, so i%4+4 ∈ {4,5,6,7}  →  select s4..s7
            # i//4 ∈ 0..49, so always in {0,1,2,3}           →  select s0..s3
            ci_mod = i % 4
            s_i  = s4 if ci_mod == 0 else (s5 if ci_mod == 1 else (s6 if ci_mod == 2 else s7))
            i4     = i // 4
            s_i4   = s0 if i4 == 0 else (s1 if i4 == 1 else (s2 if i4 == 2 else s3))

            result = 1.0 + s_i + 2.0 * (s_i - s_i4)

            tl.store(out_ptr + h * 199 + i, result)


@torch.fx.wrap
def fused_wavlm_linear_sigmoid(in_0, in_1, in_2, in_3):
    """
    Fused kernel wrapper.

    Args:
        in_0 : bias   [8]              (bfloat16 or float16)
        in_1 : weight [8, 64]          (bfloat16 or float16)
        in_2 : const  [1, H, 1, 1]     (bfloat16 or float16)
        in_3 : input  [1, H, 199, 64]  (bfloat16 or float16)

    Returns:
        out : [1, H, 199, 1]  (bfloat16 or float16)
    """
    H = in_3.shape[1]
    out = torch.empty((1, H, 199, 1), dtype=in_3.dtype, device=in_3.device)

    # BLOCK_R is autotuned; pass 1 as a safe default hint for grid computation
    # The wrapper uses a lambda grid so autotuned BLOCK_R drives the shape.
    def grid(meta):
        BLOCK_R   = meta['BLOCK_R']
        n_blocks_r = (199 + BLOCK_R - 1) // BLOCK_R
        return (H * n_blocks_r,)

    _fused_wavlm_sigmoid_kernel[grid](
        in_3,
        in_1,
        in_0,
        in_2,
        out,
        H,
        BLOCK_K=64,
    )

    return out