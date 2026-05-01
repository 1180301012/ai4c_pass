import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: cat([in_3, in_4, tmp_3], dim=2) → sigmoid → sub(0.25) → mul(pi)
# Fixed segment sizes: L1=6400, L2=1600, L3=400 → L_total=8400
#
# Strategy: ONE kernel launch, NO @triton.autotune overhead.
# Grid: (B, ceil(8400/BLOCK_SIZE)) — 2-D, avoids integer div/mod in kernel.
# All three source segments are handled by compile-time-constant branches
# (in3_mask / in4_mask / in5_mask) whose predicates are warp-uniform for
# most blocks, so the GPU optimises away the inactive loads.
# ---------------------------------------------------------------------------

def pattern(in_3, in_4, tmp_3):
    tmp_4 = torch.cat([in_3, in_4, tmp_3], 2)
    tmp_5 = tmp_4.sigmoid()
    tmp_6 = tmp_5 - 0.25
    tmp_7 = tmp_6 * 3.141592653589793
    return tmp_7


def replacement_args(in_3, in_4, tmp_3):
    return (in_3, in_4, tmp_3)


# ---------------------------------------------------------------------------
# Triton kernel — no autotune, single launch per forward call.
# BLOCK_SIZE = 1024 is a compile-time constant.
# ---------------------------------------------------------------------------

@triton.jit
def fused_cat_sigmoid_sub_mul_kernel(
    in3_ptr,   # [B, 1, 6400]  contiguous
    in4_ptr,   # [B, 1, 1600]  contiguous
    in5_ptr,   # [B, 1,  400]  contiguous (conv2d output, viewed)
    out_ptr,   # [B, 1, 8400]  output
    B,
    BLOCK_SIZE: tl.constexpr,
):
    batch = tl.program_id(0)          # in [0, B)
    bid   = tl.program_id(1)          # in [0, N_BLOCKS)

    pos   = bid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # global positions

    # Warp-uniform segment masks  (6400-aligned → no within-warp divergence
    # except at the 2 boundary blocks out of 9 total)
    in3_mask = (pos < 6400)
    in4_mask = (pos >= 6400) & (pos < 8000)
    in5_mask = (pos >= 8000) & (pos < 8400)
    valid    = (pos < 8400)

    # Safe source indices — clamped to 0 when mask is False
    idx3 = tl.where(in3_mask, batch * 6400 + pos,         0)
    idx4 = tl.where(in4_mask, batch * 1600 + (pos - 6400), 0)
    idx5 = tl.where(in5_mask, batch * 400  + (pos - 8000), 0)

    x3 = tl.load(in3_ptr + idx3, mask=in3_mask, other=0.0)
    x4 = tl.load(in4_ptr + idx4, mask=in4_mask, other=0.0)
    x5 = tl.load(in5_ptr + idx5, mask=in5_mask, other=0.0)

    val = x3 + x4 + x5                     # exactly one source active per lane

    val_f32 = val.to(tl.float32)
    val_f32 = tl.sigmoid(val_f32)
    val_f32 = (val_f32 - 0.25) * 3.141592653589793

    tl.store(out_ptr + batch * 8400 + pos, val_f32.to(val.dtype), mask=valid)


# ---------------------------------------------------------------------------
# Python wrapper (must be decorated with @torch.fx.wrap)
# ---------------------------------------------------------------------------

_BLOCK_SIZE = 1024
_N_BLOCKS   = (8400 + _BLOCK_SIZE - 1) // _BLOCK_SIZE   # = 9


@torch.fx.wrap
def fused_cat_sigmoid_sub_mul(in_3, in_4, in_5):
    B   = in_3.shape[0]
    out = torch.empty((B, 1, 8400), dtype=in_3.dtype, device=in_3.device)

    fused_cat_sigmoid_sub_mul_kernel[(B, _N_BLOCKS)](
        in_3, in_4, in_5, out, B,
        BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=8,
    )
    return out


def replacement_func():
    return fused_cat_sigmoid_sub_mul