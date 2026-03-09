import torch
import triton
import triton.language as tl
import math

def pattern(x, weight, bias):
    tmp_7 = torch.conv2d(x, weight, bias, (4, 4), (0, 0), (1, 1), 1)
    tmp_8 = tmp_7.flatten(2)
    tmp_9 = tmp_8.transpose(1, 2)
    return tmp_9

def replacement_args(x, weight, bias):
    return (x, weight, bias)

@triton.jit
def conv2d_flatten_transpose_kernel(
    x_ptr, 
    weight_ptr, 
    bias_ptr,
    out_ptr,
    batch_size,
    in_channels,
    in_height,
    in_width,
    out_channels,
    weight_height,
    weight_width,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Each program works on one output position in the flattened output
    pid = tl.program_id(0)
    
    # Output dimensions after conv2d: [batch, out_channels, out_height, out_width]
    out_height = in_height // 4
    out_width = in_width // 4
    total_out_elements = out_height * out_width
    
    # Calculate which position in the flattened output this program handles
    flattened_hw = out_height * out_width
    total_elements = batch_size * flattened_hw * out_channels
    out_idx = pid
    if out_idx >= total_elements:
        return
    
    # Convert linear index to 3D indices: [batch, flattened_hw, channels]
    batch = out_idx // (flattened_hw * out_channels)
    flattened_pos = (out_idx % (flattened_hw * out_channels)) // out_channels
    channel = out_idx % out_channels
    
    # Convert flattened position back to spatial coordinates
    out_h = flattened_pos // out_width
    out_w = flattened_pos % out_width
    
    # Calculate the conv2d operation
    sum_val = 0.0
    if bias_ptr is not None:
        sum_val = tl.load(bias_ptr + channel)
    
    # Iterate over input channels and kernel positions
    for kc in range(0, in_channels, BLOCK_SIZE_K):
        for kh in range(weight_height):
            for kw in range(weight_width):
                # Calculate input coordinates
                in_h = out_h * 4 + kh
                in_w = out_w * 4 + kw
                
                if in_h < in_height and in_w < in_width:
                    # Load input patch and weight
                    x_base = batch * in_channels * in_height * in_width + kc * in_height * in_width + in_h * in_w
                    weight_base = channel * in_channels * weight_height * weight_width + kc * weight_height * weight_width + kh * weight_width + kw
                    
                    # Process a block of input channels
                    for k_offset in range(0, min(BLOCK_SIZE_K, in_channels - kc)):
                        x_val = tl.load(x_ptr + x_base + k_offset * in_height * in_width)
                        weight_val = tl.load(weight_ptr + weight_base + k_offset * weight_height * weight_width)
                        sum_val += x_val * weight_val
    
    # Store result
    tl.store(out_ptr + out_idx, sum_val)

@torch.fx.wrap
def fused_conv_flatten_transpose(x, weight, bias):
    # Get input dimensions
    batch_size, in_channels, in_height, in_width = x.shape
    out_channels, weight_in_channels, weight_height, weight_width = weight.shape
    
    assert in_channels == weight_in_channels, "Input channels must match weight"
    assert weight_height == 4 and weight_width == 4, "Only 4x4 kernels supported"
    
    # Output dimensions
    out_height = in_height // 4
    out_width = in_width // 4
    total_out_elements = batch_size * out_channels * out_height * out_width
    
    # Create output tensor with transposed dimensions: [batch, flattened_hw, channels]
    output = torch.empty((batch_size, out_height * out_width, out_channels), dtype=x.dtype, device=x.device)
    
    # Set up launch configuration
    BLOCK_SIZE_M = 1
    BLOCK_SIZE_N = in_channels
    BLOCK_SIZE_K = 8
    
    grid_size = (total_out_elements + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    
    # Launch kernel
    conv2d_flatten_transpose_kernel[(grid_size,)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=output,
        batch_size=batch_size,
        in_channels=in_channels,
        in_height=in_height,
        in_width=in_width,
        out_channels=out_channels,
        weight_height=weight_height,
        weight_width=weight_width,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return output

def replacement_func():
    return fused_conv_flatten_transpose