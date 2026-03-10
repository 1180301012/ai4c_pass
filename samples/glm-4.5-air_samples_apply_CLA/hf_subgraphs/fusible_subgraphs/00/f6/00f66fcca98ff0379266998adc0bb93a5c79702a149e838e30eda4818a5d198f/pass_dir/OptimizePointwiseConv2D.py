import torch
import triton
import triton.language as tl

@triton.jit
def pointwise_conv2d_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size,
    out_channels,
    in_channels, 
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one output element
    pid = tl.program_id(0)
    total_elements = batch_size * out_channels * height * width
    num_programs = tl.cdiv(total_elements, BLOCK_SIZE)
    
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    if pid >= num_programs:
        return
    
    # Calculate indices for different dimensions
    output_idx = offsets
    batch_idx = output_idx // (out_channels * height * width)
    remaining = output_idx % (out_channels * height * width)
    channel_idx = remaining // (height * width)
    spatial_idx = remaining % (height * width)
    height_idx = spatial_idx // width
    width_idx = spatial_idx % width
    
    # Load input element: [batch, in_channels, height, width]
    input_offset = batch_idx * in_channels * height * width + \
                   (spatial_idx // width) * width + spatial_idx % width
    input_val = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
    
    # Load weight element and multiply
    weight_offset = channel_idx * in_channels + batch_idx % in_channels
    weight_val = tl.load(weight_ptr + weight_offset, mask=mask, other=0.0)
    result = input_val * weight_val
    
    # Add bias
    bias_offset = channel_idx
    bias_val = tl.load(bias_ptr + bias_offset if bias_ptr is not None else 0, 
                      mask=mask, other=0.0)
    result += bias_val
    
    # Store result
    output_offset = output_idx
    tl.store(output_ptr + output_offset, result, mask=mask)

@torch.fx.wrap
def optimized_pointwise_conv2d(input, weight, bias):
    # Input shapes: [batch, in_channels, height, width]
    # Weight shapes: [out_channels, in_channels, 1, 1]
    # Bias shapes: [out_channels] or [1]
    
    batch_size, in_channels, height, width = input.shape
    out_channels = weight.shape[0]
    
    # Create output tensor
    output = torch.empty((batch_size, out_channels, height, width), 
                        dtype=input.dtype, device=input.device)
    
    # Calculate total elements and block size for Triton
    total_elements = batch_size * out_channels * height * width
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # For 1x1 conv, we can optimize by simplifying the indexing
    # Create flattened views for easier computation
    input_flat = input.view(batch_size * in_channels * height * width)
    weight_flat = weight.view(out_channels * in_channels)
    output_flat = output.view(batch_size * out_channels * height * width)
    
    if bias is not None:
        bias_flat = bias.view(out_channels)
    else:
        bias_flat = None
    
    # Launch Triton kernel
    pointwise_conv2d_kernel[(num_programs,)](
        input_flat,
        weight_flat,
        bias_flat,
        output_flat,
        batch_size * in_channels * height * width,
        out_channels,
        in_channels,
        height,
        width,
        BLOCK_SIZE
    )
    
    return output

# Pattern matching for conv2d operation
def pattern(input, weight, bias):
    return torch.conv2d(input, weight, bias, (1, 1), (0, 0), (1, 1), 1)

# Argument extraction
def replacement_args(input, weight, bias):
    return (input, weight, bias)

# Replacement function
def replacement_func():
    return optimized_pointwise_conv2d