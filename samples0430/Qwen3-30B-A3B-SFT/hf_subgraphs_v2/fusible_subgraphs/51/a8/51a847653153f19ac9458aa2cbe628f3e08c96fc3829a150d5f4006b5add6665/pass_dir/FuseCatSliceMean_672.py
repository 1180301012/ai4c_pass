import torch
import triton
import triton.language as tl


@triton.jit
def _mean_hw_kernel(
    in_ptr,
    out_ptr,
    HW,
    BLOCK_HW: tl.constexpr,
):
    """
    One program per (batch, channel) pair.
    Loads up to BLOCK_HW elements (mask handles HW < BLOCK_HW),
    accumulates in float32, stores mean.
    in_ptr layout:  [B, C, H, W] contiguous  -> element [b,c,h,w] at b*C*HW + c*HW + h*W + w
    out_ptr layout: [B, C, 1, 1] contiguous  -> element [b,c,0,0] at b*C + c = pid
    """
    pid = tl.program_id(0)
    base = pid * HW
    offsets = tl.arange(0, BLOCK_HW)
    mask = offsets < HW
    x = tl.load(in_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
    mean_val = tl.sum(x) / HW
    tl.store(out_ptr + pid, mean_val)


@torch.fx.wrap
def triton_mean_hw(x):
    """
    Drop-in replacement for x.mean((2, 3), keepdim=True).
    x shape: [B, C, H, W]
    Returns: [B, C, 1, 1] same dtype as x.
    """
    B, C, H, W = x.shape
    HW = H * W
    out_mean = torch.empty((B, C, 1, 1), dtype=x.dtype, device=x.device)
    BLOCK_HW = triton.next_power_of_2(HW)
    if BLOCK_HW < 16:
        BLOCK_HW = 16
    _mean_hw_kernel[(B * C,)](x, out_mean, HW, BLOCK_HW=BLOCK_HW)
    return out_mean


def pattern(x):
    """Match: x.mean((2, 3), keepdim=True)"""
    return x.mean((2, 3), keepdim=True)


def replacement_args(x):
    return (x,)


def replacement_func():
    return triton_mean_hw