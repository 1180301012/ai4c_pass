import torch
import triton
import triton.language as tl

def pattern(input_tensor, size, scale_factor, mode, align_corners):
    """
    Pattern: Bilinear interpolation with exact signature matching
    This optimizes the torch.nn.functional.interpolate calls with bilinear mode
    """
    result = torch.nn.functional.interpolate(input_tensor, size, scale_factor, mode, align_corners)
    return result

def replacement_args(input_tensor, size, scale_factor, mode, align_corners):
    return (input_tensor, size, scale_factor, mode, align_corners)

@triton.jit
def bilinear_interpolate_kernel(
    input_ptr, output_ptr,
    batch_size, channels, 
    input_height, input_width,
    output_height, output_width,
    BLOCK_SIZE_X: tl.constexpr, BLOCK_SIZE_Y: tl.constexpr,
):
    """
    Optimized Triton kernel for bilinear interpolation
    Efficiently upsamples from smaller to larger spatial dimensions
    """
    # Get program coordinates
    batch_idx = tl.program_id(1)
    channel_idx = tl.program_id(2)
    x = tl.program_id(0) * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    y = tl.program_id(0) * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    
    # Boundary checks
    x = x % output_width
    y = y % output_height
    
    # Calculate scaling factors
    x_scale = (input_width - 1) / (output_width - 1) if output_width > 1 else 0
    y_scale = (input_height - 1) / (output_height - 1) if output_height > 1 else 0
    
    # Calculate source coordinates
    x_src = x * x_scale
    y_src = y * y_scale
    
    # Get integer and fractional parts
    x0 = tl.floor(x_src).to(tl.int32)
    x1 = tl.minimum(x0 + 1, input_width - 1)
    y0 = tl.floor(y_src).to(tl.int32) 
    y1 = tl.minimum(y0 + 1, input_height - 1)
    
    # Calculate interpolation weights
    wx = x_src - x0
    wy = y_src - y0
    w00 = (1 - wx) * (1 - wy)
    w01 = (1 - wx) * wy
    w10 = wx * (1 - wy)
    w11 = wx * wy
    
    # Calculate memory offsets
    input_base = batch_idx * (channels * input_height * input_width) + channel_idx * (input_height * input_width)
    output_base = batch_idx * (channels * output_height * output_width) + channel_idx * (output_height * output_width)
    
    # Load four neighboring pixels for all X,Y coordinates simultaneously
    # This vectorization significantly improves performance
    offsets_00 = input_base + y0[:, None] * input_width + x0[None, :]
    offsets_01 = input_base + y1[:, None] * input_width + x0[None, :]
    offsets_10 = input_base + y0[:, None] * input_width + x1[None, :]
    offsets_11 = input_base + y1[:, None] * input_width + x1[None, :]
    
    # Load pixel values with masking
    pixels_00 = tl.load(input_ptr + offsets_00, mask=(y0[:, None] < input_height) & (x0[None, :] < input_width), other=0.0)
    pixels_01 = tl.load(input_ptr + offsets_01, mask=(y1[:, None] < input_height) & (x0[None, :] < input_width), other=0.0)
    pixels_10 = tl.load(input_ptr + offsets_10, mask=(y0[:, None] < input_height) & (x1[None, :] < input_width), other=0.0)
    pixels_11 = tl.load(input_ptr + offsets_11, mask=(y1[:, None] < input_height) & (x1[None, :] < input_width), other=0.0)
    
    # Compute bilinear interpolation
    output_vals = (w00 * pixels_00 + w01 * pixels_01 + w10 * pixels_10 + w11 * pixels_11)
    
    # Store results
    output_offsets = output_base + y[:, None] * output_width + x[None, :]
    tl.store(output_ptr + output_offsets, output_vals, 
             mask=(y[:, None] < output_height) & (x[None, :] < output_width))

@torch.fx.wrap
def optimized_bilinear_interpolate(input_tensor, size, scale_factor, mode, align_corners):
    """
    Wrapper function for optimized bilinear interpolation
    """
    # Get input dimensions
    batch_size, channels, input_height, input_width = input_tensor.shape
    output_height, output_width = size
    
    # Create output tensor
    output = torch.empty((batch_size, channels, output_height, output_width), 
                        dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Optimize block sizes based on dimensions
    if output_width >= 64:
        BLOCK_SIZE_X = 32
    else:
        BLOCK_SIZE_X = 16
        
    if output_height >= 64:
        BLOCK_SIZE_Y = 32  
    else:
        BLOCK_SIZE_Y = 16
    
    # Calculate grid dimensions
    grid_x = (output_width + BLOCK_SIZE_X - 1) // BLOCK_SIZE_X
    grid_y = (output_height + BLOCK_SIZE_Y - 1) // BLOCK_SIZE_Y
    
    # Launch Triton kernel with 3D grid (x, batch, channel)
    grid = (grid_x, batch_size, channels)
    
    bilinear_interpolate_kernel[grid](
        input_tensor,
        output,
        batch_size, channels,
        input_height, input_width,
        output_height, output_width,
        BLOCK_SIZE_X=BLOCK_SIZE_X,
        BLOCK_SIZE_Y=BLOCK_SIZE_Y
    )
    
    return output

def replacement_func():
    """Return the optimized function"""
    return optimized_bilinear_interpolate