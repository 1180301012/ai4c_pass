import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_4, in_0, in_1, in_3, in_2):
    tmp_4 = torch.nn.functional.batch_norm(in_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    return (tmp_4,)

# Argument extraction function
def replacement_args(in_4, in_0, in_1, in_3, in_2):
    return (in_4, in_0, in_1, in_3, in_2)

# Triton kernel
@triton.jit
def batch_norm_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    eps,
    BLOCK_SIZE: tl.constexpr
):
    total_elements = batch_size * channels * height * width
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    b = offsets // (channels * height * width)
    c = (offsets % (channels * height * width)) // (height * width)
    h = (offsets % (height * width)) // width
    w = offsets % width

    x = tl.load(x_ptr + offsets, mask=mask)

    mean = tl.load(running_mean_ptr + c)
    var = tl.load(running_var_ptr + c)
    weight = tl.load(weight_ptr + c)
    bias = tl.load(bias_ptr + c)

    scale = weight / tl.sqrt(var + eps)
    out_val = (x - mean) * scale + bias

    tl.store(out_ptr + offsets, out_val, mask=mask)

@torch.fx.wrap
def fused_batch_norm(x, running_mean, running_var, weight, bias):
    batch_size, channels, height, width = x.shape
    total_elements = batch_size * channels * height * width

    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty_like(x)

    batch_norm_kernel[(num_programs,)](
        x_ptr=x,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        eps=1e-05,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out

def replacement_func():
    return fused_batch_norm