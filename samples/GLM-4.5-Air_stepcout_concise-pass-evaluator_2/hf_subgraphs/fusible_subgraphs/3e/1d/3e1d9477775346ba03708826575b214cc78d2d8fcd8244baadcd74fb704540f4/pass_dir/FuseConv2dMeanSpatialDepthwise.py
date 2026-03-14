import torch
import triton
import triton.language as tl


def pattern(x, weight):
    tmp_0 = weight
    tmp_1 = torch.conv2d(x, tmp_0, None, (1, 1), (1, 1), (1, 1), 384)
    tmp_0 = None
    tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    return (tmp_1, tmp_2)


def replacement_args(x, weight):
    return (x, weight)


@triton.jit
def depthwise_conv2d_mean_kernel(
    x_ptr, weight_ptr, conv_out_ptr, mean_out_ptr,
    batch_size, in_channels, height, width,
    weight_height, weight_width,
    BLOCK_SIZE_Y: tl.constexpr, BLOCK_SIZE_X: tl.constexpr,
):
    # Program ID for batch and channel parallelism
    pid_b = tl.program_id(0)  # batch dimension
    pid_c = tl.program_id(1)  # channel dimension

    # Batch and channel indices
    b = pid_b
    c = pid_c
    
    # Load the weight for this channel (depthwise convolution)
    weight_offset = c * (weight_height * weight_width)
    weight_elements = tl.load(weight_ptr + weight_offset, mask=tl.arange(0, weight_height * weight_width) < (weight_height * weight_width))
    
    # Initialize spatial sum for this channel's convolution output
    channel_spatial_sum = 0.0
    spatial_element_count = 0
    
    # Iterate over spatial locations (simplified - using program_id for spatial parallelism in real implementation)
    # For now, each thread handles one channel and accumulates spatial statistics
    for hi in range(height):
        for wi in range(width):
            # Initialize convolution value for this spatial location
            conv_val = 0.0
            
            # Perform convolution with the 3x3 kernel (padding=1 means center kernel at each pixel)
            for kh in range(weight_height):
                for kw in range(weight_width):
                    # Input coordinates with padding (padding=1 means we extend input by 1 on each side)
                    ih = hi + kh - weight_height // 2
                    iw = wi + kw - weight_width // 2
                    
                    # Check bounds (padding=1 means valid input range is 0..height-1, 0..width-1)
                    if ih >= 0 and ih < height and iw >= 0 and iw < width:
                        # Load input element: [batch, channel, height, width] contiguous layout
                        x_offset = b * in_channels * height * width + c * height * width + ih * width + iw
                        x_val = tl.load(x_ptr + x_offset, mask=True)
                        
                        # Apply kernel weight
                        weight_idx = kh * weight_width + kw
                        conv_val += x_val * weight_elements[weight_idx]
            
            # Store convolution output
            conv_offset = b * in_channels * height * width + c * height * width + hi * width + wi
            tl.store(conv_out_ptr + conv_offset, conv_val)
            
            # Add to spatial sum for mean computation (using the convolution output)
            channel_spatial_sum += conv_val
            spatial_element_count += 1
    
    # Compute and store spatial mean for this channel
    if spatial_element_count > 0:
        mean_value = channel_spatial_sum / spatial_element_count
    else:
        mean_value = 0.0
    
    mean_offset = b * in_channels + c
    tl.store(mean_out_ptr + mean_offset, mean_value)


@torch.fx.wrap
def kernel_wrapper(x, weight):
    # Move weight to GPU if it's not already there
    if weight.device.type != 'cuda':
        weight = weight.to('cuda')
    
    # Input shapes
    batch_size, in_channels, height, width = x.shape
    weight_channels, weight_height, weight_width = weight.shape
    
    # Output shapes
    conv_out_shape = (batch_size, in_channels, height, width)
    mean_out_shape = (batch_size, in_channels, 1, 1)
    
    # Create output tensors
    conv_out = torch.zeros(conv_out_shape, dtype=x.dtype, device=x.device)
    mean_out = torch.zeros(mean_out_shape, dtype=x.dtype, device=x.device)
    
    # Flatten mean output for easier kernel access
    mean_out_flat = mean_out.view(batch_size, in_channels)
    
    # Kernel launch configuration - since this is depthwise convolution, we can achieve good parallelism
    # Each thread/block can process one channel
    BLOCK_SIZE_Y = 8  # Spatial Y block size
    BLOCK_SIZE_X = 8  # Spatial X block size
    
    # Grid configuration: (batch_size * num_warps, in_channels) for parallel processing
    # In a real implementation, we'd use more sophisticated blocking
    grid = (batch_size, in_channels)
    
    # Launch the kernel
    depthwise_conv2d_mean_kernel[grid](
        x_ptr=x,
        weight_ptr=weight,
        conv_out_ptr=conv_out,
        mean_out_ptr=mean_out_flat,
        batch_size=batch_size,
        in_channels=in_channels,
        height=height,
        width=width,
        weight_height=weight_height,
        weight_width=weight_width,
        BLOCK_SIZE_Y=BLOCK_SIZE_Y,
        BLOCK_SIZE_X=BLOCK_SIZE_X,
    )
    
    return conv_out, mean_out


def replacement_func():
    return kernel_wrapper