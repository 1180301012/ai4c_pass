import torch
import triton
import triton.language as tl


def pattern(x):
    return x.mean((2, 3), keepdim=True)


def replacement_args(x):
    return (x,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=2, num_stages=1),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=1),
    ],
    key=['HW'],
)
@triton.jit
def spatial_mean_kernel(
    input_ptr,
    output_ptr,
    HW,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    input_offset = pid * HW

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for start in range(0, HW, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < HW
        x = tl.load(input_ptr + input_offset + offsets, mask=mask, other=0.0)
        acc += x.to(tl.float32)

    mean_val = tl.sum(acc) / HW
    tl.store(output_ptr + pid, mean_val)


@torch.fx.wrap
def spatial_mean_keepdim(x):
    N, C, H, W = x.shape
    HW = H * W
    NC = N * C

    out = torch.empty((N, C, 1, 1), dtype=x.dtype, device=x.device)

    spatial_mean_kernel[(NC,)](
        input_ptr=x,
        output_ptr=out,
        HW=HW,
    )

    return out


def replacement_func():
    return spatial_mean_keepdim