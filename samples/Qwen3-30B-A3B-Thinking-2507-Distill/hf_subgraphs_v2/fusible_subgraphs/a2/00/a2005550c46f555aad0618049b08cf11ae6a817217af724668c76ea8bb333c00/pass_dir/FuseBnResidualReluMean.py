import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16, num_stages=1),
    ],
    key=['HW'],
)
@triton.jit
def triton_mean_kernel(
    x_ptr, out_ptr,
    HW,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one (n, c) pair, loops over HW elements
    pid = tl.program_id(0)
    base = pid * HW
    running_sum = 0.0
    for start in range(0, HW, BLOCK_SIZE):
        offsets = base + start + tl.arange(0, BLOCK_SIZE)
        mask = (start + tl.arange(0, BLOCK_SIZE)) < HW
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        running_sum = running_sum + tl.sum(tl.where(mask, x, 0.0))
    mean_val = running_sum / HW
    tl.store(out_ptr + pid, mean_val.to(out_ptr.dtype.element_ty))


@torch.fx.wrap
def triton_mean(x):
    N, C, H, W = x.shape
    HW = H * W
    out = torch.empty((N, C, 1, 1), dtype=x.dtype, device=x.device)
    triton_mean_kernel[(N * C,)](x, out, HW)
    return out


def pattern(x):
    return x.mean((2, 3), keepdim=True)


def replacement_args(x):
    return (x,)


def replacement_func():
    return triton_mean