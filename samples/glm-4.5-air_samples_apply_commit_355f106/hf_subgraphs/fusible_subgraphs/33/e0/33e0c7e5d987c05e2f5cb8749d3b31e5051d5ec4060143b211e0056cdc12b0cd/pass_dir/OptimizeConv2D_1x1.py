import torch
import triton
import triton.language as tl

# Pattern matching function for Conv2D operation
def pattern(input_tensor, weight_tensor, bias_tensor):
    # The actual computation pattern matches conv2d with fixed parameters
    # We don't call the function here, we just match the graph structure
    # The replacement function will handle the actual implementation
    return torch.conv2d(input_tensor, weight_tensor, bias_tensor, (1, 1), (0, 0), (1, 1), 1)

# Argument extraction function
def replacement_args(input_tensor, weight_tensor, bias_tensor):
    return (input_tensor, weight_tensor, bias_tensor)

# Optimized Conv2D kernel using Triton - simpler version
@triton.jit
def conv2d_1x1_kernel_simple(
    x_ptr,
    weight_ptr, 
    bias_ptr,
    out_ptr,
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
):
    # Get program IDs - 2D grid where each program handles one output channel and spatial position
    out_channel = tl.program_id(0)
    spatial_pos = tl.program_id(1)
    
    # Decode spatial position to h, w coordinates
    h = spatial_pos // width
    w = spatial_pos % width
    
    # Create mask to check if we're within bounds
    mask_out = out_channel < out_channels
    mask_spatial = (spatial_pos < (height * width)) & (h < height) & (w < width)
    mask = mask_out & mask_spatial
    
    # If out of bounds, return early
    if not mask:
        return
    
    # Load bias for this output channel with bounds checking
    bias_mask = mask_out
    bias = tl.load(bias_ptr + out_channel, mask=bias_mask, other=0.0)
    
    # Compute convolution sum for this spatial position
    sum_val = bias
    
    # Sum over input channels: sum_i(input[batch, i, h, w] * weight[out_channel, i])
    for in_channel in range(in_channels):
        # Calculate input offset: [batch, channel, h, w] 
        input_offset = (in_channel * height * width + spatial_pos)
        
        # Calculate weight offset: [out_channel, in_channel]
        weight_offset = (out_channel * in_channels + in_channel)
        
        # Create bounds checking masks
        input_mask = (input_offset < (in_channels * height * width))
        weight_mask = (weight_offset < (out_channels * in_channels))
        
        # Load input and weight with bounds checking
        x_val = tl.load(x_ptr + input_offset, mask=input_mask, other=0.0)
        weight_val = tl.load(weight_ptr + weight_offset, mask=weight_mask, other=0.0)
        
        # Accumulate
        sum_val += x_val * weight_val
    
    # Store result with bounds checking
    output_offset = (out_channel * height * width + spatial_pos)
    output_mask = (output_offset < (out_channels * height * width))
    tl.store(out_ptr + output_offset, sum_val, mask=output_mask)

# Kernel wrapper
@torch.fx.wrap
def optimized_conv2d_1x1(input_tensor, weight_tensor, bias_tensor):
    # Get tensor shapes
    batch_size, in_channels, height, width = input_tensor.shape
    out_channels = weight_tensor.shape[0]
    
    # Calculate output shape (for 1x1 conv with stride 1,1)
    out_height = height
    out_width = width
    
    # Create output tensor
    output = torch.empty((batch_size, out_channels, out_height, out_width), 
                       dtype=input_tensor.dtype, device=input_tensor.device)
    
    # For each batch, process the conv2d separately
    for batch_idx in range(batch_size):
        # Flatten input and output for this batch to make memory access more efficient
        flattened_input = input_tensor[batch_idx].reshape(in_channels, -1)  # [in_channels, height * width]
        flattened_output = output[batch_idx].reshape(out_channels, -1)    # [out_channels, height * width]
        
        # Reshape weights to [out_channels, in_channels] for linear access
        flattened_weights = weight_tensor.reshape(out_channels, in_channels)
        
        # Calculate 2D grid size:
        # - Dimension 0: output channels (out_channels)
        # - Dimension 1: spatial positions (height * width)
        grid_m = out_channels
        grid_n = height * width
        
        # Launch kernel for this batch using 2D grid
        conv2d_1x1_kernel_simple[(grid_m, grid_n)](
            flattened_input,          # [in_channels, height * width]
            flattened_weights,        # [out_channels, in_channels] 
            bias_tensor,              # [out_channels]
            flattened_output,         # [out_channels, height * width]
            batch_size,
            in_channels,
            out_channels,
            height,
            width
        )
    
    return output

# Replacement function
def replacement_func():
    return optimized_conv2d_1x1