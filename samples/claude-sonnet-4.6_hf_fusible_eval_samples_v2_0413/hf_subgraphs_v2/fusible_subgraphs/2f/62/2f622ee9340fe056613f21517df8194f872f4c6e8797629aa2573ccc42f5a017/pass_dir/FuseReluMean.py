import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: spatial mean over dims (2, 3) with keepdim=True
# relu(in_1, inplace=True) has NO FX node (inplace side-effect).
# The input x arriving here is already relu-applied.
# ---------------------------------------------------------------------------

def pattern(x):
    return x.mean((2, 3), keepdim=True)


def replacement_args(x):
    return (x,)


# ---------------------------------------------------------------------------
# Triton kernel: spatial mean
#   Grid: (B * C,) - one program per (batch, channel) slice
#   BLOCK_HW is chosen as next-power-of-2 >= HW in the wrapper,
#   so no loop is needed – single-pass load + tl.sum.
# ---------------------------------------------------------------------------

@triton.jit
def spatial_mean_kernel(
    x_ptr,
    out_ptr,
    HW,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_HW)
    mask = offs < HW
    x = tl.load(x_ptr + pid * HW + offs, mask=mask, other=0.0)
    s = tl.sum(x.to(tl.float32), axis=0) / HW
    tl.store(out_ptr + pid, s)


@torch.fx.wrap
def triton_spatial_mean(x):
    B, C, H, W = x.shape
    BC = B * C
    HW = H * W

    # Next power of 2 >= HW (no loop needed in kernel, only masking)
    bk = 1
    while bk < HW:
        bk <<= 1
    num_warps = max(1, min(bk >> 5, 8))  # bk/32, capped at 8

    out = torch.empty((BC,), dtype=x.dtype, device=x.device)

    spatial_mean_kernel[(BC,)](
        x_ptr=x,
        out_ptr=out,
        HW=HW,
        BLOCK_HW=bk,
        num_warps=num_warps,
    )

    return out.view(B, C, 1, 1)


def replacement_func():
    return triton_spatial_mean