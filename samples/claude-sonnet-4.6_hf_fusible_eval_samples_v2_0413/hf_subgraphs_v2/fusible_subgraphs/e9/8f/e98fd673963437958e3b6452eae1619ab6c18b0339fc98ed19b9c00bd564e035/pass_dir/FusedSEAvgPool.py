import torch
import triton
import triton.language as tl


def pattern(in_1, in_2):
    """
    Match: sigmoid(in_2) -> view(1,-1,1,1) -> expand_as(in_1) -> in_1 * expanded
    Both in_1 and in_2 are model-level input placeholders.
    """
    tmp_0 = in_2.sigmoid()
    tmp_1 = tmp_0.view(1, -1, 1, 1)
    tmp_2 = tmp_1.expand_as(in_1)
    tmp_3 = in_1 * tmp_2
    return tmp_3


def replacement_args(in_1, in_2):
    return (in_1, in_2)


@triton.jit
def se_scale_kernel(
    x1_ptr, x2_ptr, out_ptr,
    C, HW,
    BLOCK_HW: tl.constexpr,
):
    """
    One program per channel.  Sigma computed once; data loaded contiguously.
    Grid: (C,)  -- 2048 programs with num_warps=8 gives full A30 occupancy.
    x1  : [1, C, H*W]  channel c starts at c*HW
    x2  : [1, 1, C]    element c at offset c
    out : same as x1
    """
    c = tl.program_id(0)

    # One sigmoid per channel (only C exp() total — same as original)
    sig_raw = tl.load(x2_ptr + c).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-sig_raw))

    # Contiguous load / multiply / store for all HW positions
    base = c * HW
    hw_offsets = tl.arange(0, BLOCK_HW)
    mask = hw_offsets < HW

    x1 = tl.load(x1_ptr + base + hw_offsets, mask=mask, other=0.0)
    out = (x1.to(tl.float32) * sig).to(x1.dtype)
    tl.store(out_ptr + base + hw_offsets, out, mask=mask)


@torch.fx.wrap
def fused_se_scale(in_1, in_2):
    """
    Replacement for: sigmoid(in_2) -> view(1,-1,1,1) -> expand_as(in_1) -> mul
    in_1: [1, C, H, W],  in_2: [1, 1, C]
    returns: [1, C, H, W]  == in_1 * sigmoid(in_2)
    """
    C = in_1.shape[1]
    HW = in_1.shape[2] * in_1.shape[3]

    out = torch.empty_like(in_1)

    # BLOCK_HW=256 covers all HW values (max 144).
    # num_warps=8 → 2048*8 = 16384 warps > 6912 max on A30 → full occupancy.
    se_scale_kernel[(C,)](
        in_1, in_2, out,
        C, HW,
        BLOCK_HW=256,
        num_warps=8,
    )

    return out


def replacement_func():
    return fused_se_scale