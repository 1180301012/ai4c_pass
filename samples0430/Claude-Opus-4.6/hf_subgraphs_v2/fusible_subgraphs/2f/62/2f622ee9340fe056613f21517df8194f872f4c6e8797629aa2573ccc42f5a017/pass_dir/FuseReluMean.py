import torch
import triton
import triton.language as tl


def pattern(in_1):
    tmp_3 = in_1.mean((2, 3), keepdim=True)
    return tmp_3


def replacement_args(in_1):
    return (in_1,)


@triton.jit
def mean_kernel(
    input_ptr,
    mean_output_ptr,
    spatial_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    base_offset = pid * spatial_size

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < spatial_size

    x = tl.load(input_ptr + base_offset + offsets, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)
    sum_val = tl.sum(x_f32)
    mean_val = sum_val / spatial_size

    tl.store(mean_output_ptr + pid, mean_val.to(x.dtype))


@torch.fx.wrap
def fused_mean(in_1):
    B = in_1.shape[0]
    C = in_1.shape[1]
    H = in_1.shape[2]
    W = in_1.shape[3]
    spatial_size = H * W
    num_channels = B * C

    mean_output = torch.empty((B, C, 1, 1), dtype=in_1.dtype, device=in_1.device)

    BLOCK_SIZE = triton.next_power_of_2(spatial_size)
    if BLOCK_SIZE < 16:
        BLOCK_SIZE = 16

    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    elif BLOCK_SIZE <= 32:
        num_warps = 1
    elif BLOCK_SIZE <= 128:
        num_warps = 2

    mean_kernel[(num_channels,)](
        in_1,
        mean_output,
        spatial_size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    return mean_output


def replacement_func():
    return fused_mean