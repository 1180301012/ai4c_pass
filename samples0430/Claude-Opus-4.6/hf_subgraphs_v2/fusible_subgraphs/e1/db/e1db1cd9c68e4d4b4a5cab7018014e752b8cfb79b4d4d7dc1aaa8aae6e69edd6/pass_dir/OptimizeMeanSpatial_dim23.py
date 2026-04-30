import torch
import triton
import triton.language as tl


def pattern(x):
    return x.mean((2, 3), keepdim=True)


def replacement_args(x):
    return (x,)


@triton.jit
def mean_spatial_kernel(
    in_ptr,
    out_ptr,
    spatial_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    base_offset = pid * spatial_size

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < spatial_size

    # Load with other=0.0 so masked elements don't affect sum
    x = tl.load(in_ptr + base_offset + offsets, mask=mask, other=0.0)

    # Accumulate in float32 for precision, multiply by reciprocal
    x_f32 = x.to(tl.float32)
    sum_val = tl.sum(x_f32, axis=0)
    mean_val = sum_val / spatial_size

    # Store result
    tl.store(out_ptr + pid, mean_val.to(x.dtype))


@torch.fx.wrap
def triton_mean_spatial(x):
    B, C, H, W = x.shape
    spatial_size = H * W
    n = B * C

    out = torch.empty((B, C, 1, 1), dtype=x.dtype, device=x.device)

    # Use different num_warps based on problem size
    if n >= 8192:
        nw = 8
    else:
        nw = 4

    mean_spatial_kernel[(n,)](
        x,
        out,
        spatial_size,
        BLOCK_SIZE=4096,
        num_warps=nw,
        num_stages=1,
    )

    return out


def replacement_func():
    return triton_mean_spatial