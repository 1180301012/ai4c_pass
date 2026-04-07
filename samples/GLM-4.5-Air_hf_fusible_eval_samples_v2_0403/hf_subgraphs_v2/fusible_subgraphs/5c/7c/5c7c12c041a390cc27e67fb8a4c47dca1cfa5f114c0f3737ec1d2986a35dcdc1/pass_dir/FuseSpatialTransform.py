import torch
import triton
import triton.language as tl

def pattern(x, target_size=(64, 64), kernel_size=2, stride=2, padding=0, ceil_mode=False, return_indices=False):
    # Max pool 2D operation
    tmp_4 = torch.nn.functional.max_pool2d(
        x, kernel_size, stride, padding, ceil_mode, return_indices
    )
    # Interpolate operation - interpolate to a specific target size
    tmp_5 = torch.nn.functional.interpolate(
        tmp_4, target_size, None, 'bilinear', False
    )
    return tmp_5  # Only return the final observable output

def replacement_args(x, target_size=(64, 64), kernel_size=2, stride=2, padding=0, ceil_mode=False, return_indices=False):
    return (x, target_size, kernel_size, stride, padding, ceil_mode, return_indices)

@triton.jit
def maxpool2d_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    in_channels,
    in_height, 
    in_width,
    kernel_size,
    stride,
    padding,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
):
    # Each program processes one spatial block
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    
    # Calculate output spatial dimensions
    out_height = (in_height + 2 * padding - kernel_size) // stride + 1
    out_width = (in_width + 2 * padding - kernel_size) // stride + 1
    
    # Calculate input coordinates for this block
    start_h = pid_h * BLOCK_SIZE_Y * stride - padding
    start_w = pid_w * BLOCK_SIZE_X * stride - padding
    
    # Process each channel in this block
    for c in range(in_channels):
        for bh in range(BLOCK_SIZE_Y):
            for bw in range(BLOCK_SIZE_X):
                out_h = pid_h * BLOCK_SIZE_Y + bh
                out_w = pid_w * BLOCK_SIZE_X + bw
                
                # Check if within output bounds
                if out_h < out_height and out_w < out_width:
                    max_val = -float('inf')
                    
                    # Process the kernel region
                    for kh in range(kernel_size):
                        for kw in range(kernel_size):
                            in_h = start_h + bh * stride + kh
                            in_w = start_w + bw * stride + kw
                            
                            # Check if within input bounds
                            if (padding <= in_h < in_height - padding and 
                                padding <= in_w < in_width - padding):
                                offset = (
                                    batch_size * in_channels * in_height * in_width +
                                    c * in_height * in_width +
                                    in_h * in_width +
                                    in_w
                                )
                                val = tl.load(input_ptr + offset)
                                if val > max_val:
                                    max_val = val
                    
                    # Store the max value
                    out_offset = (
                        batch_size * in_channels * out_height * out_width +
                        c * out_height * out_width +
                        out_h * out_width +
                        out_w
                    )
                    tl.store(output_ptr + out_offset, max_val)

@triton.jit
def interpolate_bilinear_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    in_channels,
    in_height, 
    in_width,
    out_height,
    out_width,
    # Scale factor
    scale_h: tl.constexpr,
    scale_w: tl.constexpr,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
):
    # Each program processes one pixel in the output
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    
    # Calculate corresponding input coordinates (bilinear interpolation)
    src_h = pid_h * scale_h
    src_w = pid_w * scale_w
    
    # Get the four neighboring pixels in input
    h1 = int(src_h)
    h2 = min(h1 + 1, in_height - 1)
    w1 = int(src_w)
    w2 = min(w1 + 1, in_width - 1)
    
    # Calculate interpolation weights
    wy = src_h - h1
    wx = src_w - w1
    
    # Process each channel
    for c in range(in_channels):
        # Load four neighboring pixels
        def get_pixel_val(h, w):
            offset = (
                batch_size * in_channels * in_height * in_width +
                c * in_height * in_width +
                h * in_width +
                w
            )
            return tl.load(input_ptr + offset)
        
        # Bilinear interpolation
        val00 = get_pixel_val(h1, w1)
        val01 = get_pixel_val(h1, w2) 
        val10 = get_pixel_val(h2, w1)
        val11 = get_pixel_val(h2, w2)
        
        # Interpolate along width first
        val0 = val00 * (1 - wx) + val01 * wx
        val1 = val10 * (1 - wx) + val11 * wx
        
        # Interpolate along height
        interpolated_val = val0 * (1 - wy) + val1 * wy
        
        # Store the result
        out_offset = (
            batch_size * in_channels * out_height * out_width +
            c * out_height * out_width +
            pid_h * out_width +
            pid_w
        )
        tl.store(output_ptr + out_offset, interpolated_val)

@torch.fx.wrap
def fused_spatial_transform(x, target_size=(64, 64), kernel_size=2, stride=2, padding=0, ceil_mode=False, return_indices=False):
    """Fused max_pool2d + bilinear interpolate operations"""
    batch_size, in_channels, in_height, in_width = x.shape
    
    # Max pooling: input_size/2 -> output_size/4
    out_height_pooled = (in_height + 2 * padding - kernel_size) // stride + 1
    out_width_pooled = (in_width + 2 * padding - kernel_size) // stride + 1
    
    # Create output for max pool
    maxpooled_out = torch.empty((batch_size, in_channels, out_height_pooled, out_width_pooled), 
                              dtype=x.dtype, device=x.device)
    
    # Launch max pool kernel
    BLOCK_SIZE_X = 16
    BLOCK_SIZE_Y = 16
    grid_x = (out_width_pooled + BLOCK_SIZE_X - 1) // BLOCK_SIZE_X
    grid_y = (out_height_pooled + BLOCK_SIZE_Y - 1) // BLOCK_SIZE_Y
    grid = (grid_y, grid_x)
    
    maxpool2d_kernel[grid](
        input_ptr=x,
        output_ptr=maxpooled_out,
        batch_size=batch_size,
        in_channels=in_channels,
        in_height=in_height,
        in_width=in_width,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        BLOCK_SIZE_X=BLOCK_SIZE_X,
        BLOCK_SIZE_Y=BLOCK_SIZE_Y,
    )
    
    # Use the provided target size
    target_height, target_width = target_size
    
    interpolated_out = torch.empty((batch_size, in_channels, target_height, target_width), 
                                  dtype=x.dtype, device=x.device)
    
    # Launch interpolate kernel
    scale_h = in_height / target_height if target_height > 0 else 1.0
    scale_w = in_width / target_width if target_width > 0 else 1.0
    
    BLOCK_SIZE_X = 8
    BLOCK_SIZE_Y = 8
    grid_x = (target_width + BLOCK_SIZE_X - 1) // BLOCK_SIZE_X
    grid_y = (target_height + BLOCK_SIZE_Y - 1) // BLOCK_SIZE_Y
    grid = (grid_y, grid_x)
    
    interpolate_bilinear_kernel[grid](
        input_ptr=maxpooled_out,
        output_ptr=interpolated_out,
        batch_size=batch_size,
        in_channels=in_channels,
        in_height=out_height_pooled,
        in_width=out_width_pooled,
        out_height=target_height,
        out_width=target_width,
        scale_h=scale_h,
        scale_w=scale_w,
        BLOCK_SIZE_X=BLOCK_SIZE_X,
        BLOCK_SIZE_Y=BLOCK_SIZE_Y,
    )
    
    return interpolated_out

def replacement_func():
    return fused_spatial_transform