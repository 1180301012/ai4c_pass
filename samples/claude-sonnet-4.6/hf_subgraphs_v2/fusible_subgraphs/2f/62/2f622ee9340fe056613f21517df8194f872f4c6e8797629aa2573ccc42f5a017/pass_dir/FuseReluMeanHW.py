import torch
import triton
import triton.language as tl


def pattern(in_1):
    tmp_3 = in_1.mean((2, 3), keepdim=True)
    return tmp_3


def replacement_args(in_1):
    return (in_1,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 16}, num_warps=1, num_stages=1),
        triton.Config({'BLOCK_HW': 32}, num_warps=2, num_stages=1),
        triton.Config({'BLOCK_HW': 64}, num_warps=2, num_stages=1),
        triton.Config({'BLOCK_HW': 128}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_HW': 256}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_HW': 512}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_HW': 1024}, num_warps=16, num_stages=1),
        triton.Config({'BLOCK_HW': 2048}, num_warps=16, num_stages=1),
        triton.Config({'BLOCK_HW': 4096}, num_warps=16, num_stages=1),
    ],
    key=['BC', 'HW'],
)
@triton.jit
def mean_hw_kernel(
    input_ptr,
    mean_out_ptr,
    BC,
    HW,
    BLOCK_HW: tl.constexpr,
):
    """
    One program per (B*C) slice. Scalar float32 accumulator,
    tl.sum per block → avoids large register vectors.
    Stores result directly to output dtype (no extra conversion kernel).
    """
    bc_idx = tl.program_id(0)
    in_base = bc_idx * HW
    acc = 0.0

    for block_start in range(0, HW, BLOCK_HW):
        offsets = block_start + tl.arange(0, BLOCK_HW)
        mask = offsets < HW
        x = tl.load(input_ptr + in_base + offsets, mask=mask, other=0.0)
        acc += tl.sum(x.to(tl.float32), axis=0)

    mean_val = acc / HW
    # Triton auto-converts float32 → output dtype when storing
    tl.store(mean_out_ptr + bc_idx, mean_val)


@torch.fx.wrap
def mean_hw(x):
    B, C, H, W = x.shape
    BC = B * C
    HW = H * W

    # Allocate with correct final shape and dtype — no extra conversion needed
    mean_out = torch.empty((B, C, 1, 1), dtype=x.dtype, device=x.device)

    mean_hw_kernel[(BC,)](
        x, mean_out,
        BC, HW,
    )

    return mean_out


def replacement_func():
    return mean_hw