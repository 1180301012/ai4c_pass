import torch
import triton
import triton.language as tl

def pattern(conv_input, conv_weight):
    # Start with the conv2d operation that matches original
    conv_bias = None
    conv_stride = (1, 1)
    conv_padding = (1, 1)
    conv_dilation = (1, 1)
    conv_groups = 384
    
    # Perform conv2d exactly as in original
    conv_output = torch.conv2d(conv_input, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, conv_groups)
    
    # Then compute mean reduction
    mean_dims = (2, 3)
    keepdim = True
    mean_output = conv_output.mean(mean_dims, keepdim=keepdim)
    
    return (conv_output, mean_output)

def replacement_args(input_tensor, weight):
    return (input_tensor, weight)

@triton.jit
def fused_conv2d_mean_kernel(
    input_ptr,
    weight_ptr,
    conv_output_ptr,
    mean_output_ptr,
    n_channels,
    input_height,
    input_width,
    weight_height,
    weight_width,
    BLOCK_SIZE_CHANNELS: tl.constexpr,
    BLOCK_SIZE_SPATIAL: tl.constexpr
):
    # Each program handles a subset of channels
    channel_idx = tl.program_id(0)
    
    # Calculate channel range for this program
    channel_start = channel_idx * BLOCK_SIZE_CHANNELS
    channel_end = min(channel_start + BLOCK_SIZE_CHANNELS, n_channels)
    
    # Handle mean reduction accumulation
    acc_mean = 0.0
    
    # Process spatial grid
    spatial_grid_y = tl.program_id(1) * BLOCK_SIZE_SPATIAL
    spatial_grid_x = tl.program_id(2) * BLOCK_SIZE_SPATIAL
    
    # Calculate spatial bounds
    spatial_grid_y_end = min(spatial_grid_y + BLOCK_SIZE_SPATIAL, input_height)
    spatial_grid_x_end = min(spatial_grid_x + BLOCK_SIZE_SPATIAL, input_width)
    
    # Channel offsets
    input_channel_stride = input_height * input_width
    weight_channel_stride = weight_height * weight_width
    conv_channel_stride = input_height * input_width
    
    # Process all spatial locations for this channel subset
    for h_idx in range(spatial_grid_y, spatial_grid_y_end):
        for w_idx in range(spatial_grid_x, spatial_grid_x_end):
            # For each channel in this workblock
            for c in range(channel_start, channel_end):
                # Input offset for current channel, spatial location
                input_offset = c * input_channel_stride + h_idx * input_width + w_idx
                
                # Weight offset for current channel
                weight_offset = c * weight_channel_stride
                
                # Load input and weight values
                input_val = tl.load(input_ptr + input_offset, mask=True)
                
                # Load and apply weights (3x3 kernel)
                conv_val = 0.0
                for kh in range(weight_height):
                    for kw in range(weight_width):
                        weight_offset_kw = weight_offset + kh * weight_width + kw
                        input_offset_kw = input_offset + kh * input_width + kw
                        
                        # Handle boundary conditions
                        if (h_idx + kh) < input_height and (w_idx + kw) < input_width:
                            conv_val += tl.load(weight_ptr + weight_offset_kw, mask=True) * \
                                      tl.load(input_ptr + input_offset_kw, mask=True)
                
                # Store conv output
                conv_output_offset = c * conv_channel_stride + h_idx * input_width + w_idx
                tl.store(conv_output_ptr + conv_output_offset, conv_val)
                
                # Accumulate for mean (only for the first spatial location per channel)
                if h_idx == spatial_grid_y and w_idx == spatial_grid_x:
                    acc_mean += conv_val
    
    # Compute mean for this channel subset
    num_spatial = input_height * input_width
    mean_val = acc_mean / num_spatial
    
    # Store mean results
    for c in range(channel_start, channel_end):
        mean_output_offset = c
        tl.store(mean_output_ptr + mean_output_offset, mean_val)

def fused_kernel_wrapper(input_tensor, weight_tensor):
    # Get tensor shapes
    n_channels = input_tensor.shape[1]
    input_height = input_tensor.shape[2]
    input_width = input_tensor.shape[3]
    
    weight_height = weight_tensor.shape[2]
    weight_width = weight_tensor.shape[3]
    
    # Output shapes
    conv_output_shape = input_tensor.shape
    mean_output_shape = (n_channels, 1, 1)
    
    # Create output tensors
    conv_output = torch.empty(conv_output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    mean_output = torch.empty(mean_output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Configure kernel launch
    BLOCK_SIZE_CHANNELS = 32  # Reduced block size for better GPU utilization
    BLOCK_SIZE_SPATIAL = 8    # Spatial block size
    
    # Calculate grid dimensions
    grid_x = (input_width + BLOCK_SIZE_SPATIAL - 1) // BLOCK_SIZE_SPATIAL
    grid_y = (input_height + BLOCK_SIZE_SPATIAL - 1) // BLOCK_SIZE_SPATIAL
    grid_z = (n_channels + BLOCK_SIZE_CHANNELS - 1) // BLOCK_SIZE_CHANNELS
    
    # Launch kernel
    fused_conv2d_mean_kernel[(grid_z, grid_y, grid_x)](
        input_tensor,
        weight_tensor,
        conv_output,
        mean_output,
        n_channels,
        input_height,
        input_width,
        weight_height,
        weight_width,
        BLOCK_SIZE_CHANNELS,
        BLOCK_SIZE_SPATIAL
    )
    
    return conv_output, mean_output

@torch.fx.wrap
def optimized_fused_operation(input_tensor, weight_tensor):
    return fused_kernel_wrapper(input_tensor, weight_tensor)

def replacement_func():
    return optimized_fused_operation