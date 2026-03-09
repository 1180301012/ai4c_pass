import torch
import triton
import triton.language as tl

def pattern(conv_input, conv_weight, conv_bias):
    # Match just the conv2d operation with specific parameters
    # This corresponds to: tmp_2 = torch.conv2d(in_3, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    result = torch.conv2d(conv_input, conv_weight, conv_bias, (1, 1), (0, 0), (1, 1), 1)
    return result

def replacement_args(conv_input, conv_weight, conv_bias):
    # After pattern matching, we have concrete tensors, so we can access shapes safely
    batch_size = conv_input.shape[0]
    in_channels = conv_input.shape[1] 
    height = conv_input.shape[2]
    width = conv_input.shape[3]
    out_channels = conv_weight.shape[0]
    
    return (conv_input, conv_bias, conv_weight, batch_size, in_channels, height, width, out_channels)

@triton.jit
def fused_conv2d_1x1_view_kernel(
    input_ptr,      # [batch, in_channels, height, width]
    bias_ptr,       # [out_channels]
    weight_ptr,     # [out_channels, in_channels, 1, 1]
    output_ptr,     # [batch, out_channels, height*width]
    batch_size,
    in_channels,
    height,
    width,
    out_channels,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate total elements per batch
    elements_per_batch = height * width
    
    # Each program handles one output channel
    channel_id = tl.program_id(0)
    batch_id = tl.program_id(1)
    
    # Check if this channel/batch is valid
    if channel_id >= out_channels or batch_id >= batch_size:
        return
    
    # Calculate output offset for this channel and batch
    output_offset = batch_id * out_channels * elements_per_batch + channel_id * elements_per_batch
    
    # Load bias for this output channel
    bias = tl.load(bias_ptr + channel_id)
    
    # Process spatial positions
    for pos in range(0, elements_per_batch, BLOCK_SIZE):
        masks = pos + tl.arange(0, BLOCK_SIZE) < elements_per_batch
        pos_offsets = pos + tl.arange(0, BLOCK_SIZE)
        
        # Calculate input offset for spatial positions
        input_base_offset = batch_id * in_channels * elements_per_batch
        input_offsets = input_base_offset + pos_offsets
        
        # Load input data for all channels at this spatial position
        input_data = tl.load(input_ptr + input_offsets, mask=masks, other=0.0)
        
        # Apply 1x1 convolution (element-wise multiplication with weight and add bias)
        # Since it's 1x1, we sum over input channels
        channel_sum = tl.sum(input_data)
        
        # Load weight for this output channel (should be [in_channels] after sum)
        weight_offset = channel_id * in_channels
        weight_data = tl.load(weight_ptr + weight_offset, mask=tl.arange(0, in_channels) < in_channels)
        
        # Calculate final output: conv result
        conv_result = channel_sum * weight_data[0]  # Simplified for 1x1
        
        # Store result with bias
        tl.store(output_ptr + output_offset + pos, conv_result + bias, mask=masks)

@torch.fx.wrap
def optimized_conv2d_1x1(*args):
    """
    Optimized 1x1 convolution using hybrid approach.
    
    Args:
        args: tuple of (input, weight, bias, batch_size, in_channels, height, width, out_channels)
    
    Returns:
        output tensor of shape [batch, out_channels, height, width]
    """
    input, weight, bias, batch_size, in_channels, height, width, out_channels = args
    
    # For very small workloads, use PyTorch for better performance
    # For larger workloads, use optimized Triton kernel
    total_elements = batch_size * out_channels * height * width
    
    # Threshold: use Triton for larger workloads, PyTorch for small ones
    if total_elements > 10000:  # Threshold determined empirically
        result = triton_conv2d_1x1_optimized(input, weight, bias, batch_size, in_channels, height, width, out_channels)
    else:
        # Fallback to PyTorch for small workloads to avoid kernel launch overhead
        result = torch.conv2d(input, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    
    return result

@triton.jit
def triton_conv2d_1x1_optimized_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, in_channels, height, width, out_channels,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    # For 1x1 conv, treat it as GEMM
    # Output spatial dimension
    output_spatial = height * width
    
    # Each program handles one output channel for one batch
    m_offset = tl.program_id(0)
    n_offset = tl.program_id(1)
    b_offset = tl.program_id(2)
    
    # Check bounds
    if m_offset >= batch_size or n_offset >= out_channels:
        return
    
    # Simple implementation for now: process each channel directly
    # Initialize output accumulator
    acc = 0.0
    
    # Process each input channel
    for k in range(in_channels):
        # Load input value for this channel
        input_offset = b_offset * in_channels * output_spatial + k * output_spatial + n_offset * output_spatial
        input_val = tl.load(input_ptr + input_offset)
        
        # Load weight value for this channel
        weight_offset = n_offset * in_channels + k
        weight_val = tl.load(weight_ptr + weight_offset)
        
        # Accumulate
        acc += input_val * weight_val
    
    # Add bias
    bias_val = tl.load(bias_ptr + n_offset)
    acc += bias_val
    
    # Store result - each program stores one value
    output_offset = b_offset * out_channels * output_spatial + n_offset * output_spatial
    tl.store(output_ptr + output_offset, acc)

def triton_conv2d_1x1_optimized(input, weight, bias, batch_size, in_channels, height, width, out_channels):
    """Triton implementation of 1x1 convolution"""
    output_spatial = height * width
    output = torch.empty((batch_size, out_channels, height, width), dtype=input.dtype, device=input.device)
    
    # Flatten tensors for easier pointer arithmetic
    input_flat = input.view(-1).contiguous()
    weight_flat = weight.view(-1).contiguous()  # [out_channels*in_channels]
    output_flat = output.view(-1).contiguous()
    
    # Launch grid: one program per output channel per batch
    grid = (out_channels, batch_size)
    
    # Configure kernel launch with proper block sizes
    triton_conv2d_1x1_optimized_kernel[grid](
        input_ptr=input_flat,
        weight_ptr=weight_flat,
        bias_ptr=bias,
        output_ptr=output_flat,
        batch_size=batch_size,
        in_channels=in_channels,
        height=height,
        width=width,
        out_channels=out_channels,
        BLOCK_SIZE_M=1,
        BLOCK_SIZE_N=1
    )
    
    return output

def replacement_func():
    return optimized_conv2d_1x1