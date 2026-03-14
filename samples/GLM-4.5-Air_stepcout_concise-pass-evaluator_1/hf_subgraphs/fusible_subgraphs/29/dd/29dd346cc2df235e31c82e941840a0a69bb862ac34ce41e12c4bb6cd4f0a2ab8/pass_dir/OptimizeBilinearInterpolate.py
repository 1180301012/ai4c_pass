import torch
import triton
import triton.language as tl

# Pattern matching function - matches individual interpolate operations
def pattern(x, size=None, scale_factor=None, mode='bilinear', align_corners=False, recompute_scale_factor=None):
    """
    Match torch.nn.functional.interpolate with bilinear mode
    """
    result = torch.nn.functional.interpolate(x, size=size, scale_factor=scale_factor, mode=mode, 
                                          align_corners=align_corners, recompute_scale_factor=recompute_scale_factor)
    return result

# Argument extraction function
def replacement_args(x, size=None, scale_factor=None, mode='bilinear', align_corners=False, recompute_scale_factor=None):
    return (x, size, scale_factor, mode, align_corners, recompute_scale_factor)

# Optimized kernel for bilinear interpolation
@triton.jit
def interpolate_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    channels,
    in_height,
    in_width,
    out_height,
    out_width,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Optimized bilinear interpolation kernel
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Calculate output coordinates
    out_y = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    out_x = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks for valid pixels
    mask_m = out_y < out_height
    mask_n = out_x < out_width
    mask = mask_m[:, None] & mask_n[None, :]
    
    # Calculate scale factors
    if in_height > 0 and out_height > 0:
        y_scale = (in_height - 1) / max(out_height - 1, 1)
    else:
        y_scale = 1.0
        
    if in_width > 0 and out_width > 0:
        x_scale = (in_width - 1) / max(out_width - 1, 1)
    else:
        x_scale = 1.0
    
    # Convert output coordinates to input coordinates
    in_y_f = out_y * y_scale
    in_x_f = out_x * x_scale
    
    # Get integer parts and fractional parts
    in_y0 = tl.math.floor(in_y_f).to(tl.int32)
    in_x0 = tl.math.floor(in_x_f).to(tl.int32)
    in_y1 = tl.math.ceil(in_y_f).to(tl.int32)
    in_x1 = tl.math.ceil(in_x_f).to(tl.int32)
    
    # Clamp to valid range
    in_y0 = tl.maximum(0, tl.minimum(in_height - 1, in_y0))
    in_x0 = tl.maximum(0, tl.minimum(in_width - 1, in_x0))
    in_y1 = tl.maximum(0, tl.minimum(in_height - 1, in_y1))
    in_x1 = tl.maximum(0, tl.minimum(in_width - 1, in_x1))
    
    # Calculate weights
    y_frac = in_y_f - in_y0
    x_frac = in_x_f - in_x0
    
    # Handle boundary cases for align_corners=False (default)
    if in_height == 1:
        y_frac = 0.0
        in_y0 = 0
        in_y1 = 0
    if in_width == 1:
        x_frac = 0.0
        in_x0 = 0
        in_x1 = 0
    
    # Compute bilinear weights
    w00 = (1 - y_frac) * (1 - x_frac)
    w01 = (1 - y_frac) * x_frac
    w10 = y_frac * (1 - x_frac)
    w11 = y_frac * x_frac
    
    # Load four corners for all pixels in this block
    # Batch dimension loop
    for b in range(0, batch_size, 1):
        # Channel dimension loop
        for c in range(0, channels, 1):
            # Base pointer for this batch and channel
            base_input = input_ptr + (b * channels + c) * in_height * in_width
            base_output = output_ptr + (b * channels + c) * out_height * out_width
            
            # Load corner values
            val00 = tl.load(base_input + in_y0[:, None] * in_width + in_x0[None, :], mask=mask, other=0.0)
            val01 = tl.load(base_input + in_y0[:, None] * in_width + in_x1[None, :], mask=mask, other=0.0)
            val10 = tl.load(base_input + in_y1[:, None] * in_width + in_x0[None, :], mask=mask, other=0.0)
            val11 = tl.load(base_input + in_y1[:, None] * in_width + in_x1[None, :], mask=mask, other=0.0)
            
            # Compute bilinear interpolation
            interpolated = (w00 * val00 + w01 * val01 + w10 * val10 + w11 * val11)
            
            # Store result
            tl.store(base_output + out_y[:, None] * out_width + out_x[None, :], interpolated, mask=mask)

@torch.fx.wrap
def optimized_interpolate(x, size=None, scale_factor=None, mode='bilinear', align_corners=False, recompute_scale_factor=None):
    """
    Optimized bilinear interpolation using Triton
    """
    if mode != 'bilinear':
        # For non-bilinear modes, fall back to PyTorch
        return torch.nn.functional.interpolate(x, size=size, scale_factor=scale_factor, mode=mode,
                                              align_corners=align_corners, recompute_scale_factor=recompute_scale_factor)
    
    # Get input shape
    batch_size, channels, in_height, in_width = x.shape
    
    # Determine output size
    if size is not None:
        out_height, out_width = size
    elif scale_factor is not None:
        if isinstance(scale_factor, (list, tuple)):
            sf_h, sf_w = scale_factor
        else:
            sf_h = sf_w = scale_factor
        out_height = int(in_height * sf_h)
        out_width = int(in_width * sf_w)
    else:
        out_height, out_width = in_height, in_width
    
    # Create output tensor
    output = torch.empty((batch_size, channels, out_height, out_width), dtype=x.dtype, device=x.device)
    
    # Special case: identity operation (same input and output size)
    if (out_height == in_height) and (out_width == in_width):
        output.copy_(x)
        return output
    
    # Set up kernel grid
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    
    # Adjust grid size for optimal GPU occupancy
    grid_m = (out_height + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (out_width + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch kernel
    interpolate_kernel[(grid_m, grid_n, 1)](
        x,
        output,
        batch_size,
        channels,
        in_height,
        in_width,
        out_height,
        out_width,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N
    )
    
    return output

# Replacement function (returns function reference)  
def replacement_func():
    return optimized_interpolate