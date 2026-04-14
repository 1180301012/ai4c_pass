import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: matches the full scale-bias-cat normalisation in model.py
# ---------------------------------------------------------------------------
def pattern(in_0, in_1):
    tmp_1 = in_1 * 0.458
    tmp_2 = tmp_1 + -0.030000000000000027
    tmp_3 = in_0[(slice(None, None, None), 1)]
    tmp_4 = torch.unsqueeze(tmp_3, 1)
    tmp_5 = tmp_4 * 0.448
    tmp_6 = tmp_5 + -0.08799999999999997
    tmp_7 = in_0[(slice(None, None, None), 2)]
    tmp_8 = torch.unsqueeze(tmp_7, 1)
    tmp_9 = tmp_8 * 0.45
    tmp_10 = tmp_9 + -0.18799999999999994
    tmp_11 = torch.cat((tmp_2, tmp_6, tmp_10), 1)
    return tmp_11


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ---------------------------------------------------------------------------
# Triton kernel – 1-D grid over all B*H*W elements.
#
# Design:
#  • 1-D grid: each program handles BLOCK_SIZE consecutive flat indices.
#    b_idx = offset // HW   (fast bit-shift when HW is power-of-2)
#    hw_idx = offset % HW   (fast bitmask when HW is power-of-2)
#  • No @triton.autotune.  BLOCK_SIZE is chosen deterministically at Python
#    level (see wrapper), so there is NO interference-corrupted config
#    selection, and NO autotune warmup that can be inherited by later cases.
#  • For n_elements ≤ 2048 the wrapper picks BLOCK_SIZE = n_elements, giving
#    exactly ONE program per kernel launch.  A single program never suffers
#    the "two programs run serially on an overloaded GPU" penalty.
# ---------------------------------------------------------------------------
@triton.jit
def _fused_norm_cat_kernel(
    in0_ptr,     # [B, C_in, H, W] contiguous
    in1_ptr,     # [B, 1,    H, W] contiguous
    out_ptr,     # [B, 3,    H, W] contiguous
    HW,          # H * W  (runtime scalar)
    C_in,        # input channel count  (runtime scalar)
    n_elements,  # B * H * W  (runtime scalar)
    BLOCK_SIZE: tl.constexpr,
):
    pid     = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask    = offsets < n_elements

    # Recover batch index and spatial position within the batch
    b_idx  = offsets // HW   # fast bit-shift when HW is a power-of-2
    hw_idx = offsets % HW    # fast bitmask when HW is a power-of-2

    # ---- Channel 0: in_1[:, 0, :, :] ----
    in1_val = tl.load(in1_ptr + b_idx * HW + hw_idx, mask=mask, other=0.0)
    ch0     = in1_val * 0.458 + (-0.030000000000000027)

    # ---- Channel 1: in_0[:, 1, :, :] ----
    in0_ch1_val = tl.load(in0_ptr + b_idx * C_in * HW + HW + hw_idx,
                          mask=mask, other=0.0)
    ch1 = in0_ch1_val * 0.448 + (-0.08799999999999997)

    # ---- Channel 2: in_0[:, 2, :, :] ----
    in0_ch2_val = tl.load(in0_ptr + b_idx * C_in * HW + 2 * HW + hw_idx,
                          mask=mask, other=0.0)
    ch2 = in0_ch2_val * 0.45 + (-0.18799999999999994)

    # ---- Store to [B, 3, H, W] ----
    out_base = b_idx * 3 * HW + hw_idx
    tl.store(out_ptr + out_base,          ch0, mask=mask)
    tl.store(out_ptr + out_base + HW,     ch1, mask=mask)
    tl.store(out_ptr + out_base + 2 * HW, ch2, mask=mask)


# ---------------------------------------------------------------------------
# Host wrapper
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_norm_cat(in_0, in_1):
    B, C_in, H, W = in_0.shape
    HW         = H * W
    n_elements = B * HW

    out = torch.empty((B, 3, H, W), dtype=in_0.dtype, device=in_0.device)

    # Deterministic BLOCK_SIZE selection:
    #   • Choose the smallest power-of-2 that fits n_elements (up to 2048).
    #   • For n_elements ≤ 2048 this gives ceil(n_elements/BLOCK_SIZE) = 1
    #     program → never serialises on a shared GPU under heavy load.
    #   • No autotune → no interference-corrupted config selection.
    if n_elements <= 256:
        BLOCK_SIZE, NW = 256, 4
    elif n_elements <= 512:
        BLOCK_SIZE, NW = 512, 4
    elif n_elements <= 1024:
        BLOCK_SIZE, NW = 1024, 4
    else:
        BLOCK_SIZE, NW = 2048, 8   # 8 elems/thread vectorisation for large HW

    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    _fused_norm_cat_kernel[grid](
        in_0, in_1, out,
        HW, C_in, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=NW,
    )

    return out


def replacement_func():
    return fused_norm_cat