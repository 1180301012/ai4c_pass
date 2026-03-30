import torch
import triton
import triton.language as tl
import math

@triton.jit
def hardsigmoid_kernel(x, out, n_elements):
    """Hardsigmoid activation function computed in Triton"""
    pid = tl.program_id(0)
    block_size = tl.num_programs(0)
    offsets = pid * block_size + tl.arange(0, block_size)
    
    mask = offsets < n_elements
    x_val = tl.load(x + offsets, mask=mask)
    
    # hardsigmoid: max(0, min(1, 0.2 * x + 0.5))
    result = tl.where(
        x_val < -2.5,
        0.0,
        tl.where(
            x_val > 5.0,
            1.0,
            0.2 * x_val + 0.5
        )
    )
    
    tl.store(out + offsets, result, mask=mask)

# Pattern matching function
def pattern(in_3, in_1, in_0, in_2):
    """
    Match the computation pattern:
    conv2d + hardsigmoid + element-wise multiplication
    """
    conv_result = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    hardsigmoid_result = torch.nn.functional.hardsigmoid(conv_result, False)
    multiply_result = in_2 * hardsigmoid_result
    return multiply_result

# Argument extraction function
def replacement_args(in_3, in_1, in_0, in_2):
    return (in_3, in_1, in_0, in_2)

@triton.jit
def fused_conv_hardsigmoid_multiply_kernel(
    x_ptr,           # in_3: [batch, channels, 1, 1]
    weight_ptr,      # in_1: [out_channels, in_channels, 1, 1]
    bias_ptr,        # in_0: [out_channels]
    scale_ptr,       # in_2: [batch, channels, height, width]
    out_ptr,         # output: [batch, out_channels, height, width]
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
):
    """Simplified kernel that handles one batch-channel pair completely"""
    # Grid setup: each program handles one batch and one output channel
    batch_idx = tl.program_id(0)      # batch dimension
    out_channel_idx = tl.program_id(1)  # output channel dimension
    
    # Load weights and bias (only need to do this once per program)
    weight_offset = out_channel_idx * in_channels
    weight = tl.load(weight_ptr + weight_offset)
    bias = tl.load(bias_ptr + out_channel_idx)
    
    # Load input channel attention (same for all spatial positions)
    input_attention = tl.load(x_ptr + batch_idx * in_channels)
    
    # Compute conv result
    conv_result = input_attention * weight + bias
    
    # Apply hardsigmoid
    hardsigmoid_val = tl.where(
        conv_result < -2.5,
        0.0,
        tl.where(
            conv_result > 5.0,
            1.0,
            0.2 * conv_result + 0.5
        )
    )
    
    # Process all spatial positions in a single loop
    for h_pos in range(height):
        for w_pos in range(width):
            # Load scale value
            scale_offset = (batch_idx * out_channels * height * width + 
                           out_channel_idx * height * width + 
                           h_pos * width + w_pos)
            scale_val = tl.load(scale_ptr + scale_offset)
            
            # Apply multiplication
            result = scale_val * hardsigmoid_val
            
            # Store output
            output_offset = (batch_idx * out_channels * height * width + 
                            out_channel_idx * height * width + 
                            h_pos * width + w_pos)
            tl.store(out_ptr + output_offset, result)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def fused_conv_hardsigmoid_multiply(in_3, in_1, in_0, in_2):
    # Get tensor shapes and dimensions
    batch_size = in_2.shape[0]
    spatial_channels = in_2.shape[1]
    spatial_height = in_2.shape[2]
    spatial_width = in_2.shape[3]
    
    # Note: in_3 has shape [batch, channels, 1, 1] where channels = in_channels
    batch_size_input = in_3.shape[0]
    in_channels = in_3.shape[1]
    
    # Note: in_1 has shape [out_channels, in_channels, 1, 1]
    out_channels = in_1.shape[0]
    
    # Validate shapes
    if in_channels != spatial_channels:
        raise ValueError(f"Input channels mismatch: in_channels={in_channels}, spatial_channels={spatial_channels}")
    if batch_size_input != batch_size:
        raise ValueError(f"Batch size mismatch: input_batch={batch_size_input}, spatial_batch={batch_size}")
    
    # Calculate grid dimensions - 2D grid: batch x output_channel
    grid_x = batch_size
    grid_y = out_channels
    
    # Create output tensor with same shape as input_2
    out = torch.empty_like(in_2)
    
    # Launch kernel with 2D grid
    fused_conv_hardsigmoid_multiply_kernel[(
        grid_x, 
        grid_y,
    )](
        in_3,
        in_1,
        in_0,
        in_2,
        out,
        batch_size,
        in_channels,
        out_channels,
        spatial_height,
        spatial_width,
    )
    
    return out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_conv_hardsigmoid_multiply