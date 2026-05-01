import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: per-channel scale+bias followed by cat on dim=1
# Matches the operations in model.py (googlenet_start2_end13_1) exactly.
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
# Triton kernel  –  2-D grid  (B, ceil(HW / BLOCK))
#   axis-0  = batch index
#   axis-1  = spatial tile
#
# Each CTA processes one batch element × BLOCK spatial positions across
# all three output channels in a single pass, avoiding intermediate tensors
# and reducing memory traffic by ~60%.
# ---------------------------------------------------------------------------

@triton.jit
def _fused_norm_cat_kernel(
    in0_ptr,            # [B, C_in0, H, W] contiguous
    in1_ptr,            # [B, 1,     H, W] contiguous
    out_ptr,            # [B, 3,     H, W] contiguous
    C_in0,
    HW,
    BLOCK: tl.constexpr,
):
    b    = tl.program_id(0)
    tile = tl.program_id(1)
    offs = tile * BLOCK + tl.arange(0, BLOCK)
    mask = offs < HW

    # Channel 0 from in_1[:,0,:,:]  (scale + bias interleaved with next load)
    val0 = tl.load(in1_ptr + b * HW + offs, mask=mask, other=0.0)
    val0 = val0 * 0.458 + (-0.030000000000000027)

    # Channels 1 & 2 from in_0[:,1,:,:] and in_0[:,2,:,:]
    in0_base = b * C_in0 * HW
    val1 = tl.load(in0_ptr + in0_base + 1 * HW + offs, mask=mask, other=0.0)
    val1 = val1 * 0.448 + (-0.08799999999999997)
    val2 = tl.load(in0_ptr + in0_base + 2 * HW + offs, mask=mask, other=0.0)
    val2 = val2 * 0.45  + (-0.18799999999999994)

    # Write all 3 output channels
    out_base = b * 3 * HW
    tl.store(out_ptr + out_base + 0 * HW + offs, val0, mask=mask)
    tl.store(out_ptr + out_base + 1 * HW + offs, val1, mask=mask)
    tl.store(out_ptr + out_base + 2 * HW + offs, val2, mask=mask)


# ---------------------------------------------------------------------------
# Config table  (threshold, BLOCK, num_warps)
#
# Design notes for NVIDIA A30 (56 SMs, Ampere):
#  * Small HW (≤1024): one tile per batch item, BLOCK fits all elements in
#    one CTA.  num_warps=4 → 128 threads → 8 elements/thread (BLOCK=1024).
#  * Large HW (>1024): BLOCK=2048 with num_warps=8 (256 threads, 4 elem/thread).
#    For HW=50176 → 25 CTAs.  Although SM occupancy is low (0.44 CTAs/SM),
#    each CTA is large enough to hide memory latency independently.
# ---------------------------------------------------------------------------
_CFG = [
    (64,    64,   1),
    (128,   128,  2),
    (256,   256,  2),
    (512,   512,  4),
    (1024,  1024, 4),
    (None,  2048, 8),
]

def _select_cfg(HW):
    for thresh, block, nw in _CFG:
        if thresh is None or HW <= thresh:
            return block, nw
    return 2048, 8


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_norm_cat(in_0, in_1):
    B     = in_0.shape[0]
    C_in0 = in_0.shape[1]
    H     = in_0.shape[2]
    W     = in_0.shape[3]
    HW    = H * W

    out = torch.empty((B, 3, H, W), dtype=in_0.dtype, device=in_0.device)

    BLOCK, nw = _select_cfg(HW)
    n_tiles   = (HW + BLOCK - 1) // BLOCK

    _fused_norm_cat_kernel[(B, n_tiles)](
        in_0, in_1, out,
        C_in0, HW,
        BLOCK=BLOCK,
        num_warps=nw,
    )

    return out


def replacement_func():
    return fused_norm_cat