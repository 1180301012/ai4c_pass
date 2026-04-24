import torch
import triton
import triton.language as tl


def pattern(tmp_0):
    tmp_1 = tmp_0.mean((2, 3))
    return tmp_1


def replacement_args(tmp_0):
    return (tmp_0,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64},   num_warps=2),
        triton.Config({'BLOCK_SIZE': 128},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['HW'],
)
@triton.jit
def mean_hw_kernel(
    x_ptr,
    out_ptr,
    HW,
    BLOCK_SIZE: tl.constexpr,
):
    """One program per (batch, channel). Reduces over the H*W dimension."""
    pid = tl.program_id(0)
    base = pid * HW
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for start in range(0, HW, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < HW
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        acc = acc + tl.where(mask, x.to(tl.float32), 0.0)

    tl.store(out_ptr + pid, tl.sum(acc, axis=0) / HW)


@torch.fx.wrap
def triton_mean_hw(x):
    """Triton mean over spatial dims (H, W) of a [B, C, H, W] tensor."""
    B, C, H, W = x.shape
    HW = H * W
    out = torch.empty((B, C), dtype=x.dtype, device=x.device)
    mean_hw_kernel[(B * C,)](x, out, HW)
    return out


def replacement_func():
    return triton_mean_hw