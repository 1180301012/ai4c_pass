import torch
import triton
import triton.language as tl

def pattern(in_9, in_4):
    """Match grouped convolution + view optimization pattern"""
    # Grouped convolution with 512 groups - must match exactly with keyword args
    conv2d = torch.conv2d(input=in_9, weight=in_4, groups=512)
    # View operation that can be optimized out
    tmp_5 = conv2d.view(1, 512, 64, 64)
    return conv2d, tmp_5

def replacement_args(in_9, in_4):
    """Extract arguments for optimized depthwise convolution"""
    return in_9, in_4

@triton.jit
def optimized_depthwise_conv_kernel(
    output_ptr,
    input_ptr,
    weight_ptr,
    batch_size,
    in_channels,
    in_height,
    in_width,
    out_height,
    out_width,
    kernel_size,
    stride: tl.constexpr,
    padding: tl.constexpr,
    dilation: tl.constexpr,
    groups: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized depthwise convolution kernel for grouped convolutions"""
    # Get program ID and calculate total elements to process
    pid = tl.program_id(0)
    n_elements = batch_size * groups * out_height * out_width
    n_programs = tl.cdiv(n_elements, BLOCK_SIZE)
    
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Calculate indices
    batch_idx = offsets // (groups * out_height * out_width)
    group_idx = (offsets // (out_height * out_width)) % groups
    out_h_idx = (offsets // out_width) % out_height
    out_w_idx = offsets % out_width
    
    # Calculate input indices with padding
    in_h_idx = out_h_idx * stride - padding
    in_w_idx = out_w_idx * stride - padding
    
    # Only process valid spatial locations
    valid_mask = (in_h_idx >= 0) & (in_h_idx < in_height) & (in_w_idx >= 0) & (in_w_idx < in_width)
    mask = mask & valid_mask
    
    # Load kernel values (assuming 1x7x7 kernel, spatial part is 1x7x7 for each group)
    # For depthwise conv with groups=in_channels, each group has kernel_size[0] x kernel_size[1] x 1
    kernel_offsets = group_idx * (kernel_size * kernel_size) + tl.arange(0, kernel_size * kernel_size)
    kernel_vals = tl.load(weight_ptr + kernel_offsets, mask=True, other=0.0).to(tl.float32)
    
    # Load input values for current kernel locations
    in_h_positions = in_h_idx.unsqueeze(0) + tl.arange(0, kernel_size)
    in_w_positions = in_w_idx.unsqueeze(0) + tl.arange(0, kernel_size)
    
    # Reshape kernel and broadcast positions
    kernel_vals = kernel_vals.reshape(kernel_size, kernel_size)
    in_h_positions = in_h_positions.reshape(1, kernel_size)
    in_w_positions = in_w_positions.reshape(kernel_size, 1)
    
    # Calculate input pointer offsets
    batch_offset = batch_idx * in_channels * in_height * in_width
    group_offset = group_idx * in_height * in_width
    h_offset = in_h_positions * in_width + in_w_positions
    
    # Convert to linear indices
    input_offsets = batch_offset + group_offset + h_offset
    
    # Load input values
    input_vals = tl.load(input_ptr + input_offsets.flatten(), mask=mask.any(), other=0.0)
    
    # Reshape for matrix multiplication
    input_vals = input_vals.reshape(kernel_size, kernel_size)
    
    # Apply convolution (simple dot product)
    conv_result = tl.sum(input_vals * kernel_vals)
    
    # Store result
    output_idx = offsets
    tl.store(output_ptr + output_idx, conv_result.to(tl.bfloat16), mask=mask)

@torch.fx.wrap
def optimized_depthwise_conv(input_tensor, weight_tensor):
    """Optimized depthwise convolution wrapper"""
    input_shape = input_tensor.shape
    weight_shape = weight_tensor.shape
    
    batch_size, in_channels, in_height, in_width = input_shape
    out_channels, kernel_height, kernel_width = weight_shape[0], weight_shape[2], weight_shape[3]
    groups = in_channels  # For depthwise conv, groups = in_channels
    
    # Calculate output dimensions
    out_height = (in_height + 2*1 - kernel_height) // 1 + 1  # stride=1, padding=1
    out_width = (in_width + 2*1 - kernel_width) // 1 + 1
    
    # Calculate total elements and configure grid
    n_elements = batch_size * groups * out_height * out_width
    BLOCK_SIZE = 256  # Optimized for depthwise conv
    n_programs = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create output tensor
    output = torch.empty((batch_size, groups, out_height, out_width), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch kernel
    optimized_depthwise_conv_kernel[(n_programs,)](
        output_ptr=output,
        input_ptr=input_tensor,
        weight_ptr=weight_tensor,
        batch_size=batch_size,
        in_channels=in_channels,
        in_height=in_height,
        in_width=in_width,
        out_height=out_height,
        out_width=out_width,
        kernel_size=kernel_height,
        stride=1,
        padding=1,
        dilation=1,
        groups=groups,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    """Return the optimized depthwise convolution function"""
    return optimized_depthwise_conv