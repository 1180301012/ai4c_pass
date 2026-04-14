import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    """
    Pattern for conv2d + gelu (before dropout)
    Matches the structure before dropout elimination: conv2d → gelu
    """
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (1, 1), (1, 1), in_0.shape[0])
    tmp_3 = torch.nn.functional.gelu(conv2d)
    return tmp_3

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def fused_conv2d_gelu_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, in_channels, in_height, in_width, out_channels,
    BLOCK_SIZE_Y: tl.constexpr, BLOCK_SIZE_X: tl.constexpr
):
    # Get program IDs  
    pid_y = tl.program_id(0)  # Batch dimension
    pid_x = tl.program_id(1)  # Channel dimension
    
    # Compute ranges for current program
    y_offsets = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    x_offsets = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    
    # Create masks
    y_mask = y_offsets < batch_size
    x_mask = x_offsets < out_channels
    
    # For depthwise conv2d with groups=out_channels, each input channel i uses weight[i]
    # and processes spatial locations independently
    
    # Compute linear indices for input and output
    input_idx = (y_offsets[:, None] * in_channels * in_height * in_width + 
                x_offsets[None, :] * in_height * in_width)
    output_idx = (y_offsets[:, None] * out_channels * in_height * in_width + 
                 x_offsets[None, :] * in_height * in_width)
    
    # Load input values - [batch_size, out_channels, height, width]
    input_vals = tl.load(input_ptr + input_idx, mask=y_mask[:, None] & x_mask[None, :], other=0.0)
    
    # Load bias values - one per output channel
    bias_vals = tl.load(bias_ptr + x_offsets, mask=x_mask, other=0.0)
    
    # Load weight values - for depthwise conv, weights are [out_channels, 1, 1]
    weight_vals = tl.load(weight_ptr + x_offsets, mask=x_mask, other=0.0)

    # For depthwise conv2d with 1x1 kernel, it's element-wise multiplication + bias per channel
    # Apply: input * weight + bias
    conv_out = input_vals * weight_vals[None, :, None, None] + bias_vals[None, :, None, None]
    
    # Flatten for GELU computation
    conv_flat = conv_out.reshape(-1)
    
    # Apply GELU approximation: x * 0.5 * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    gelu_out = conv_flat * 0.5 * (1.0 + tl.tanh(tl.sqrt(2.0 / 3.14159) * (conv_flat + 0.044715 * conv_flat ** 3)))
    
    # Store result
    tl.store(output_ptr + output_idx, gelu_out.reshape(input_vals.shape), mask=y_mask[:, None] & x_mask[None, :])

@torch.fx.wrap
def fused_depthwise_conv2d_gelu(input_tensor, weight_tensor, bias_tensor):
    """Fused depthwise conv2d + GELU operation"""
    
    # Get tensor shapes
    batch_size, in_channels, in_height, in_width = input_tensor.shape
    out_channels = bias_tensor.shape[0]  # For depthwise conv, groups = out_channels
    
    # Ensure output tensor has correct shape
    output = torch.empty((batch_size, out_channels, in_height, in_width), 
                        dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Choose block sizes based on tensor dimensions
    BLOCK_SIZE_Y = min(64, batch_size)
    BLOCK_SIZE_X = min(256, out_channels)
    
    # Calculate grid
    grid_y = (batch_size + BLOCK_SIZE_Y - 1) // BLOCK_SIZE_Y
    grid_x = (out_channels + BLOCK_SIZE_X - 1) // BLOCK_SIZE_X
    grid = (grid_y, grid_x)
    
    # Launch kernel
    fused_conv2d_gelu_kernel[grid](
        input_ptr=input_tensor,
        weight_ptr=weight_tensor,
        bias_ptr=bias_tensor,
        output_ptr=output,
        batch_size=batch_size,
        in_channels=in_channels,
        in_height=in_height,
        in_width=in_width,
        out_channels=out_channels,
        BLOCK_SIZE_Y=BLOCK_SIZE_Y,
        BLOCK_SIZE_X=BLOCK_SIZE_X
    )
    
    return output

def replacement_func():
    return fused_depthwise_conv2d_gelu