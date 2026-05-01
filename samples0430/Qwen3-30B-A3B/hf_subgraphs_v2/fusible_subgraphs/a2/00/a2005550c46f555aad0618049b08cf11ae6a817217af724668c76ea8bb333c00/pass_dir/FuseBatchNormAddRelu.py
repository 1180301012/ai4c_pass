import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_4, in_0, in_1, in_3, in_2, in_5):
    tmp_4 = torch.nn.functional.batch_norm(in_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_5 = in_5 + tmp_4
    tmp_6 = torch.nn.functional.relu(tmp_5, inplace=False)
    return (tmp_6,)

# Argument extraction function
def replacement_args(in_4, in_0, in_1, in_3, in_2, in_5):
    return (in_4, in_0, in_1, in_3, in_2, in_5)

# Triton kernel
def fused_bn_add_relu_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    residual_ptr,
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

    # Convert flattened index to 4D coordinates
    b = offsets // (channels * height * width)
    c = (offsets % (channels * height * width)) // (height * width)
    h = (offsets % (height * width)) // width
    w = offsets % width

    # Load input x
    x = tl.load(x_ptr + offsets, mask=mask)

    # Load channel-wise statistics
    mean = tl.load(running_mean_ptr + c)
    var = tl.load(running_var_ptr + c)
    weight = tl.load(weight_ptr + c)
    bias = tl.load(bias_ptr + c)

    # Compute the scale: weight / sqrt(var + eps)
    scale = weight / tl.sqrt(var + eps)

    # Compute residual at this spatial position
    residual = tl.load(residual_ptr + offsets, mask=mask)

    # Compute the result: max(0, (x - mean)*scale + bias + residual)
    temp = (x - mean) * scale + bias + residual
    out_val = tl.maximum(temp, 0.0)

    # Store the result
    tl.store(out_ptr + offsets, out_val, mask=mask)

# Wrapper function (must be decorated with @torch.fx.wrap)
@torch.fx.wrap
def fused_bn_add_relu(x, running_mean, running_var, weight, bias, residual):
    # Get the input shape
    batch_size, channels, height, width = x.shape
    total_elements = batch_size * channels * height * width

    # Set the block size
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Create output tensor
    out = torch.empty_like(x)

    # Launch the kernel
    fused_bn_add_relu_kernel[(num_programs,)](
        x_ptr=x,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        residual_ptr=residual,
        out_ptr=out,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        eps=1e-05,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out

# Replacement function
def replacement_func():
    return fused_bn_add_relu