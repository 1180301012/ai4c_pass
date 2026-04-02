import torch
import triton
import triton.language as tl

def pattern(input_tensor, weight_tensor, bias_tensor):
    """Pattern matching for 1x1 pointwise convolution"""
    conv2d_output = torch.conv2d(input_tensor, weight_tensor, bias_tensor, (1, 1), (0, 0), (1, 1), 1)
    return conv2d_output

def replacement_args(input_tensor, weight_tensor, bias_tensor):
    return (input_tensor, weight_tensor, bias_tensor)

@triton.jit
def optimized_pointwise_conv_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size, 
    input_channels,
    output_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel for 1x1 pointwise convolution"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (batch_size * input_channels * output_size)
    
    # Compute indices
    batch_idx = offsets // (input_channels * output_size)
    channel_idx = (offsets % (input_channels * output_size)) // output_size
    spatial_idx = offsets % output_size
    
    # Load input (each channel has the same value across all spatial positions for 1x1 conv)
    input_val = tl.load(input_ptr + batch_idx * input_channels * output_size + channel_idx * output_size + spatial_idx, mask=mask)
    
    # Load weight (we only need the weight for this input channel)
    weight_val = tl.load(weight_ptr + channel_idx * 1 + spatial_idx // (output_size // (20 * 20)), mask=mask)
    
    # Load bias
    bias_val = tl.load(bias_ptr, mask=tl.arange(0, 1) < 1)
    
    # Compute output: input * weight + bias
    result = input_val * weight_val + bias_val
    
    tl.store(output_ptr + offsets, result, mask=mask)

@triton.jit
def optimized_pointwise_conv_kernel_simple(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size,
    input_channels,
    output_channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel for 1x1 convolution with correct channel handling"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (batch_size * output_channels * height * width)
    
    # Compute indices from linear offset for output tensor
    batch_idx = offsets // (output_channels * height * width)
    output_channel_idx = (offsets % (output_channels * height * width)) // (height * width)
    spatial_idx = offsets % (height * width)
    
    # Initialize accumulator as an array (vector) to match the type of operations
    total_val = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Sum over all input channels
    for input_channel_idx in range(input_channels):
        # Load input value for this channel - use indexing for all BLOCK_SIZE elements
        input_offsets = (batch_idx * input_channels + input_channel_idx) * height * width + spatial_idx
        input_vals = tl.load(input_ptr + input_offsets, mask=input_channel_idx < input_channels)
        
        # Load weight for this output channel and input channel
        weight_offset = output_channel_idx * input_channels + input_channel_idx
        weight_vals = tl.load(weight_ptr + weight_offset)
        
        # Accumulate: input * weight (broadcast)
        total_val += input_vals * weight_vals
    
    # Load bias - broadcast to block size
    bias_vals = tl.load(bias_ptr + output_channel_idx)
    
    # Compute final output
    result = total_val + bias_vals
    
    # Store output
    output_offset = offsets
    tl.store(output_ptr + output_offset, result, mask=mask)

@torch.fx.wrap
def optimized_pointwise_conv2d(input_tensor, weight_tensor, bias_tensor):
    """Optimized 1x1 pointwise convolution"""
    batch_size, input_channels, height, width = input_tensor.shape
    
    # Get weight tensor shape for output channels
    output_channels = weight_tensor.shape[0]
    
    output = torch.empty((batch_size, output_channels, height, width), 
                        dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Use block size optimized for better GPU occupancy
    block_size = 256  # Smaller block size for better performance with inner loop
    
    total_elements = batch_size * output_channels * height * width
    num_programs = (total_elements + block_size - 1) // block_size
    
    # Reshape weights to [output_channels, input_channels] for easier access
    weight_2d = weight_tensor.reshape(output_channels, input_channels)
    
    optimized_pointwise_conv_kernel_simple[(num_programs,)](
        input_ptr=input_tensor,
        weight_ptr=weight_2d,
        bias_ptr=bias_tensor,
        output_ptr=output,
        batch_size=batch_size,
        input_channels=input_channels,
        output_channels=output_channels,
        height=height,
        width=width,
        BLOCK_SIZE=block_size,
    )
    
    return output

def replacement_func():
    return optimized_pointwise_conv2d