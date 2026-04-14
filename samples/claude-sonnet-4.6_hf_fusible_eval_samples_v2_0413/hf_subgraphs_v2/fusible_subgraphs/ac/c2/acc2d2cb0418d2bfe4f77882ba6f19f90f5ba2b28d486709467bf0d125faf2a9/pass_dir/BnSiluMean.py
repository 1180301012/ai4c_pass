import torch
import triton
import triton.language as tl


def pattern(x):
    mean = x.mean((2, 3), keepdim=True)
    return mean


def replacement_args(x):
    return (x,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 32},  num_warps=2),
        triton.Config({'BLOCK_SIZE': 64},  num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
    ],
    key=['HW'],
)
@triton.jit
def mean_kernel(
    x_ptr,
    mean_out_ptr,
    C, HW,
    BLOCK_SIZE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    # Each program handles one (n, c) slice of size HW
    pid = tl.program_id(0)   # pid = n * C + c

    base = pid * HW
    acc = 0.0

    for block_start in range(0, HW, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < HW

        x = tl.load(x_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
        acc = acc + tl.sum(tl.where(mask, x, 0.0))

    mean_val = acc / HW
    tl.store(mean_out_ptr + pid, mean_val.to(OUT_DTYPE))


_DTYPE_MAP = {
    torch.float32:  tl.float32,
    torch.float16:  tl.float16,
    torch.bfloat16: tl.bfloat16,
}


@torch.fx.wrap
def mean_func(x):
    N, C, H, W = x.shape
    HW = H * W

    mean_out = torch.empty(N * C, device=x.device, dtype=x.dtype)

    out_dtype = _DTYPE_MAP.get(x.dtype, tl.float32)

    grid = (N * C,)

    mean_kernel[grid](
        x, mean_out,
        C, HW,
        OUT_DTYPE=out_dtype,
    )

    mean_out = mean_out.view(N, C, 1, 1)
    return mean_out


def replacement_func():
    return mean_func