import torch
import triton
import triton.language as tl


def pattern(x):
    return x.mean((2, 3), keepdim=True)


def replacement_args(x):
    return (x,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_HW': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_HW': 512}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_HW': 2048}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_HW': 4096}, num_warps=16, num_stages=3),
        triton.Config({'BLOCK_HW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8, num_stages=2),
    ],
    key=['HW'],
)
@triton.jit
def triton_mean_kernel(
    x_ptr, out_ptr,
    HW,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    base = x_ptr + pid * HW

    acc = tl.zeros([BLOCK_HW], dtype=tl.float32)

    for start in range(0, HW, BLOCK_HW):
        offsets = start + tl.arange(0, BLOCK_HW)
        mask = offsets < HW
        x = tl.load(base + offsets, mask=mask, other=0.0).to(tl.float32)
        acc += tl.where(mask, x, 0.0)

    mean_val = tl.sum(acc) / HW
    tl.store(out_ptr + pid, mean_val)


@torch.fx.wrap
def triton_mean(x):
    N, C, H, W = x.shape
    HW = H * W
    out = torch.empty((N, C, 1, 1), dtype=torch.float32, device=x.device)
    grid = (N * C,)
    triton_mean_kernel[grid](x, out, HW)
    return out.to(x.dtype)


def replacement_func():
    return triton_mean