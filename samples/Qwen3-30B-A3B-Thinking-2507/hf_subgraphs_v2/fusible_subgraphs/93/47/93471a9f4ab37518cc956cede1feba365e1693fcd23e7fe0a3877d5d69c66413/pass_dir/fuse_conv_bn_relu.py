import torch
import triton
import triton.language as tl

# Pattern matching function
# Matches conv2d + view + batch_norm + relu sequence
# Note: The view operation is redundant but must be matched as per original model

def pattern(in_9, in_4, in_0, in_1, in_2, in_3):
    conv = torch.conv2d(input=in_9, weight=in_4, groups=512)
    view = conv.view(1, 512, 64, 64)  # Redundant reshape, but matches model
    bn = torch.nn.functional.batch_norm(view, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    relu = torch.nn.functional.relu(bn, inplace=False)
    return relu

# Argument extraction function

def replacement_args(in_9, in_4, in_0, in_1, in_2, in_3):
    return (in_9, in_4, in_0, in_1, in_2, in_3)

# Triton kernel for fused depthwise conv + batch norm + relu
@triton.jit
def depthwise_conv_bn_relu_kernel(
    input_ptr,
    weight_ptr,
    mean_ptr,
    var_ptr,
    weight_bn_ptr,
    bias_bn_ptr,
    output_ptr,
    H_in, W_in, H_out, W_out, C,
    BLOCK_C: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr
):
    # Calculate global thread block coordinates
    h_block = tl.program_id(0)
    w_block = tl.program_id(1)
    c_block = tl.program_id(2)

    # Calculate starting position for this block
    h_start = h_block * BLOCK_H
    w_start = w_block * BLOCK_W
    c_start = c_block * BLOCK_C

    # Calculate thread offsets within block
    c = c_start + tl.thread_id(0) % BLOCK_C
    h = h_start + tl.thread_id(0) // BLOCK_C % BLOCK_H
    w = w_start + tl.thread_id(0) // BLOCK_C // BLOCK_H

    # Early exit if out of bounds
    if c >= C or h >= H_out or w >= W_out:
        return

    # Initialize convolution output
    conv_val = 0.0

    # Perform depthwise convolution (7x7 kernel)
    for kh in range(7):
        for kw in range(7):
            h_idx = h + kh
            w_idx = w + kw
            if h_idx < H_in and w_idx < W_in:
                # Load input for current channel and position
                input_val = tl.load(input_ptr + c * (H_in * W_in) + h_idx * W_in + w_idx)
                # Load weight for current channel and kernel position
                weight_val = tl.load(weight_ptr + c * 49 + kh * 7 + kw)
                conv_val += input_val * weight_val

    # Apply batch norm
    mean_val = tl.load(mean_ptr + c)
    var_val = tl.load(var_ptr + c)
    weight_bn_val = tl.load(weight_bn_ptr + c)
    bias_bn_val = tl.load(bias_bn_ptr + c)
    bn_val = (conv_val - mean_val) * tl.rsqrt(var_val + 1e-05) * weight_bn_val + bias_bn_val

    # Apply ReLU
    relu_val = tl.maximum(0.0, bn_val)

    # Store result
    output_idx = c * (H_out * W_out) + h * W_out + w
    tl.store(output_ptr + output_idx, relu_val)

# Kernel wrapper
@torch.fx.wrap
def fuse_conv_bn_relu(input, weight, mean, var, weight_bn, bias_bn):
    # Get tensor shapes
    _, C, H_in, W_in = input.shape
    H_out, W_out = 64, 64  # Based on input shape (70-6=64)

    # Allocate output
    output = torch.empty((1, C, H_out, W_out), dtype=input.dtype, device=input.device)

    # Constants for kernel configuration
    BLOCK_C = 16
    BLOCK_H = 8
    BLOCK_W = 8

    # Calculate grid dimensions
    grid_h = (H_out + BLOCK_H - 1) // BLOCK_H
    grid_w = (W_out + BLOCK_W - 1) // BLOCK_W
    grid_c = (C + BLOCK_C - 1) // BLOCK_C

    # Launch kernel
    depthwise_conv_bn_relu_kernel[
        (grid_h, grid_w, grid_c),
        BLOCK_C * BLOCK_H * BLOCK_W
    ](
        input_ptr=input,
        weight_ptr=weight,
        mean_ptr=mean,
        var_ptr=var,
        weight_bn_ptr=weight_bn,
        bias_bn_ptr=bias_bn,
        output_ptr=output,
        H_in=H_in, W_in=W_in, H_out=H_out, W_out=W_out, C=C,
        BLOCK_C=BLOCK_C, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W
    )

    return output

# Replacement function

def replacement_func():
    return fuse_conv_bn_relu