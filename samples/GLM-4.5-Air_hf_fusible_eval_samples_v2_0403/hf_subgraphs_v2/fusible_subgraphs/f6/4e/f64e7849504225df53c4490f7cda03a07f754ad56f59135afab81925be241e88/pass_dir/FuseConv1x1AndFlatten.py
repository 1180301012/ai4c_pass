import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    """
    Pattern matching for 1x1 conv2d followed by flatten operation.
    
    This matches:
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.flatten(conv2d, 2)
    """
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.flatten(conv2d, 2)
    return tmp_3

def replacement_args(in_0, in_1, in_2):
    """
    Extract arguments needed for optimized kernel.
    in_0: bias tensor [output_channels]
    in_1: weight tensor [output_channels, input_channels, 1, 1] 
    in_2: input tensor [batch_size, input_channels, height, width]
    """
    return (in_0, in_1, in_2)

@triton.jit
def fused_conv1x1_flatten_kernel(
    input_ptr,          # [batch_size, input_channels, height, width]
    weight_ptr,         # [output_channels, input_channels, 1, 1]
    bias_ptr,           # [output_channels]
    output_ptr,         # [batch_size, output_channels, height*width]
    spatial_size: tl.constexpr,
    input_channels: tl.constexpr,
    output_channels: tl.constexpr
):
    """
    Optimized kernel that fuses 1x1 conv2d + flatten into a single operation.
    
    Each program computes one spatial element for one batch and one output channel.
    """
    # Program IDs for batch, output channel, and spatial position
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    spatial_pos = tl.program_id(2)
    
    # Calculate spatial index (0 to spatial_size-1)
    spatial_idx = batch_idx * output_channels * spatial_size + channel_idx * spatial_size + spatial_pos
    
    # If we're beyond the valid spatial size, return early
    if spatial_pos >= spatial_size:
        return
    
    # Initialize output with bias
    output_val = tl.load(bias_ptr + channel_idx)
    
    # Loop over input channels
    for c_in in range(input_channels):
        # Load weight for current output and input channel
        weight_val = tl.load(weight_ptr + channel_idx * input_channels + c_in)
        
        # Load input value for current channel and spatial position
        input_idx = (batch_idx * input_channels + c_in) * spatial_size + spatial_pos
        input_val = tl.load(input_ptr + input_idx)
        
        # Multiply and accumulate
        output_val += input_val * weight_val
    
    # Store result
    tl.store(output_ptr + spatial_idx, output_val)

@torch.fx.wrap  
def fused_conv1x1_flatten_optimized(bias, weight, input_tensor):
    """
    Optimized function that fuses 1x1 conv2d + flatten using custom Triton kernel.
    """
    # Get tensor shapes and metadata
    batch_size, input_channels, height, width = input_tensor.shape
    output_channels = bias.shape[0]
    
    # Verify this is a 1x1 convolution
    assert weight.shape[2] == 1 and weight.shape[3] == 1, "Only 1x1 kernels supported"
    assert weight.shape == (output_channels, input_channels, 1, 1), "Invalid weight shape"
    
    # Create output tensor: [batch_size, output_channels, height*width]
    spatial_size = height * width
    output_shape = (batch_size, output_channels, spatial_size)
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Calculate grid dimensions: one program for each (batch, output_channel, spatial_position)
    grid = (batch_size, output_channels, spatial_size)
    
    # Launch kernel
    fused_conv1x1_flatten_kernel[grid](
        input_ptr=input_tensor,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        spatial_size=spatial_size,
        input_channels=input_channels,
        output_channels=output_channels
    )
    
    return output

def replacement_func():
    """
    Returns the optimized function that fuses conv1x1 + flatten.
    """
    return fused_conv1x1_flatten_optimized