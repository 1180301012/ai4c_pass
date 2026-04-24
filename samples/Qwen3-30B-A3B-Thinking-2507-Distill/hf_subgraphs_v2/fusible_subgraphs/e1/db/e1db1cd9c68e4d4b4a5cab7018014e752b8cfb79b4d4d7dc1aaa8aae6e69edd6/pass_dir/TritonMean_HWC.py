import torch
import triton
import triton.language as tl


def pattern(x):
    return x.mean((2, 3), keepdim=True)


def replacement_args(x):
    return (x,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 4096}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_HW': 4096}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_HW': 4096}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_HW': 4096}, num_warps=32, num_stages=2),
        triton.Config({'BLOCK_HW': 2048}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_HW': 2048}, num_warps=4,  num_stages=2),
    ],
    key=['HW'],
)
@triton.jit
def _triton_mean_hw_kernel(
    in_ptr,
    out_ptr,
    HW,
    BLOCK_HW: tl.constexpr,
):
    # One program per (batch, channel) pair
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_HW)
    mask = offsets < HW
    base = pid * HW

    # Load input elements; pad out-of-range with 0.0
    x = tl.load(in_ptr + base + offsets, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)

    # Sum valid elements (masked-out elements are 0 from other=0.0 load)
    x_sum = tl.sum(x_f32, axis=0)
    mean_val = x_sum / HW

    # Store mean (float32)
    tl.store(out_ptr + pid, mean_val)


@torch.fx.wrap
def triton_mean_hw(x):
    B, C, H, W = x.shape
    HW = H * W
    BC = B * C

    # Output shape matches PyTorch mean: [B, C, 1, 1]
    out = torch.empty((B, C, 1, 1), dtype=torch.float32, device=x.device)

    _triton_mean_hw_kernel[(BC,)](
        x,
        out,
        HW=HW,
    )
    return out


def replacement_func():
    return triton_mean_hw