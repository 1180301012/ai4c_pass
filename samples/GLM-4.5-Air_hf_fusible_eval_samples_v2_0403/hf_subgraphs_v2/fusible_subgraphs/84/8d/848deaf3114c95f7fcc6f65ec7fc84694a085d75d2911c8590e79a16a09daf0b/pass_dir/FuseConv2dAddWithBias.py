import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    """Match conv2d + dropout + addition pattern. Dropout with p=0.0 is a no-op."""
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    dropout = torch.nn.functional.dropout(conv2d, 0.0, False, False)
    result = dropout + in_2
    return result

def replacement_args(in_0, in_1, in_2, in_3):
    """Extract arguments needed for the fused conv2d + add kernel."""
    return (in_0, in_1, in_2, in_3)

@triton.jit
def fused_conv2d_add_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    add_ptr,
    output_ptr,
    spatial_size,
    batch_size,
    height,
    width,
    IN_CHANNELS: tl.constexpr,
    OUT_CHANNELS: tl.constexpr,
):
    """Fused 1x1 convolution + addition kernel."""
    
    # Each program handles one spatial position for one batch and one output channel
    pid_m = tl.program_id(0)  # batch dimension
    pid_n = tl.program_id(1)  # output channel dimension
    pid_k = tl.program_id(2)  # spatial position
    
    # Skip if out of bounds
    if pid_k >= spatial_size:
        return
    
    # Load bias for this output channel
    bias = tl.load(bias_ptr + pid_n)
    
    # Load all input channels for this spatial position
    input_offset = pid_m * IN_CHANNELS * spatial_size + pid_k * IN_CHANNELS
    input_channels = tl.arange(0, IN_CHANNELS)
    input_offset_full = input_offset + input_channels
    
    # Load input features for all channels
    input_features = tl.load(input_ptr + input_offset_full)
    
    # Load weights for this output channel and all input channels
    weight_offset = pid_n * IN_CHANNELS + input_channels
    weights = tl.load(weight_ptr + weight_offset)
    
    # Compute 1x1 convolution: sum over input_channels
    conv_result = tl.sum(input_features * weights)
    
    # Add bias
    conv_result += bias
    
    # Load addition tensor for this spatial position
    add_offset = pid_m * OUT_CHANNELS * spatial_size + pid_n * spatial_size + pid_k
    add_value = tl.load(add_ptr + add_offset)
    
    # Final result: conv + add
    final_result = conv_result + add_value
    
    # Store output
    output_offset = pid_m * OUT_CHANNELS * spatial_size + pid_n * spatial_size + pid_k
    tl.store(output_ptr + output_offset, final_result)

@torch.fx.wrap 
def fused_conv2d_add(bias, weight, add_tensor, input_tensor):
    """Wrapper function for fused conv2d + add operation."""
    
    # Get tensor shapes from the input tensor (in_3) which is the conv input
    batch_size, in_channels, height, width = input_tensor.shape
    out_channels = bias.shape[0]
    
    # Output shape should be [batch_size, out_channels, height, width]
    output = torch.empty((batch_size, out_channels, height, width), 
                        dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Calculate grid dimensions based on output tensor dimensions
    # For this kernel:
    # dim 0: batch_size (we process one batch at a time)  
    # dim 1: out_channels (we process one output channel at a time)
    # dim 2: spatial positions (we process one spatial position at a time)
    spatial_size = height * width
    
    grid = (batch_size, out_channels, spatial_size)
    
    fused_conv2d_add_kernel[grid](
        input_ptr=input_tensor,
        weight_ptr=weight,
        bias_ptr=bias,
        add_ptr=add_tensor,
        output_ptr=output,
        spatial_size=spatial_size,
        batch_size=batch_size,
        height=height,
        width=width,
        IN_CHANNELS=in_channels,
        OUT_CHANNELS=out_channels,
    )
    
    return output

def replacement_func():
    """Return the fused conv2d + add function."""
    return fused_conv2d_add