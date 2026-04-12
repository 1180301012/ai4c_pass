import torch
import triton
import triton.language as tl

# Pattern matching function - must match exact operations from model.py
def pattern(small_tensor, large_tensor):
    """Match Bilinear Interpolate followed by Element-wise multiplication"""
    # Use exact argument pattern from model.py
    # torch.nn.functional.interpolate(tmp_2, (64, 128), None, 'bilinear', False)
    # The interpolate function signature: interpolate(input, size, None, mode, align_corners) 
    interpolated = torch.nn.functional.interpolate(small_tensor, (64, 128), None, 'bilinear', False)
    result = large_tensor * interpolated
    return result

# Argument extraction function
def replacement_args(small_tensor, large_tensor):
    return (small_tensor, large_tensor)

# Triton kernel for fused bilinear interpolate + multiplication
@triton.jit
def fused_interp_mul_kernel(
    small_ptr, large_ptr, output_ptr,
    batch_size, channels, in_height, in_width, out_height, out_width,
    BLOCK_SIZE: tl.constexpr
):
    """Fused bilinear interpolate + multiplication kernel using Triton"""
    pid = tl.program_id(0)
    total_elements = batch_size * channels * out_height * out_width
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Convert offset to coordinates
    n = offsets // (channels * out_height * out_width)
    offset_c = offsets % (channels * out_height * out_width)
    c = offset_c // (out_height * out_width)
    h = (offset_c % (out_height * out_width)) // out_width
    w = offset_c % out_width
    
    # Calculate corresponding coordinate in input tensor
    # Bilinear interpolation: scale_x = out_width / in_width, scale_y = out_height / in_height
    scale_x = in_width / out_width
    scale_y = in_height / out_height
    
    # Calculate the corresponding position in input space
    src_x_float = w * scale_x
    src_y_float = h * scale_y
    
    # Get integer coordinates and fractional parts
    src_x0 = int(src_x_float)
    src_y0 = int(src_y_float)
    src_x1 = min(src_x0 + 1, in_width - 1)
    src_y1 = min(src_y0 + 1, in_height - 1)
    
    # Calculate interpolation weights
    dx = src_x_float - src_x0
    dy = src_y_float - src_y0
    
    # Bilinear interpolation
    interpolated = 0.0
    
    for c_in in range(channels):
        # Load four neighboring pixels
        idx_00 = n * channels * in_height * in_width + c_in * in_height * in_width + src_y0 * in_width + src_x0
        idx_01 = n * channels * in_height * in_width + c_in * in_height * in_width + src_y0 * in_width + src_x1
        idx_10 = n * channels * in_height * in_width + c_in * in_height * in_width + src_y1 * in_width + src_x0
        idx_11 = n * channels * in_height * in_width + c_in * in_height * in_width + src_y1 * in_width + src_x1
        
        val_00 = tl.load(small_ptr + idx_00, mask=False)
        val_01 = tl.load(small_ptr + idx_01, mask=False)
        val_10 = tl.load(small_ptr + idx_10, mask=False)
        val_11 = tl.load(small_ptr + idx_11, mask=False)
        
        # Bilinear interpolation formula
        interpolated += ((1 - dx) * (1 - dy) * val_00 +
                        dx * (1 - dy) * val_01 +
                        (1 - dx) * dy * val_10 +
                        dx * dy * val_11)
    
    # Multiply with corresponding large tensor element
    large_idx = n * channels * out_height * out_width + c * out_height * out_width + h * out_width + w
    large_val = tl.load(large_ptr + large_idx, mask=False)
    
    result = interpolated * large_val
    tl.store(output_ptr + offsets, result, mask=mask)

# Optimized kernel for very common pattern: [1, C, 1, W] -> [1, C, H, W]
@triton.jit
def optimized_1w_to_hw_interp_mul_kernel(
    small_ptr, large_ptr, output_ptr,
    channels, in_width, out_height, out_width,
    BLOCK_SIZE: tl.constexpr
):
    """Optimized kernel for [1, C, 1, W] -> [1, C, H, W] interpolate + multiply"""
    pid = tl.program_id(0)
    total_elements = channels * out_height * out_width
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Convert offset to coordinates (only 1 batch)
    offset_c = offsets
    c = offset_c // (out_height * out_width)
    h = (offset_c % (out_height * out_width)) // out_width
    w = offset_c % out_width
    
    # Since input height is 1, interpolation is simpler (1D)
    scale_x = in_width / out_width
    src_x_float = w * scale_x
    src_x0 = int(src_x_float)
    src_x1 = min(src_x0 + 1, in_width - 1)
    dx = src_x_float - src_x0
    
    # Bilinear interpolation (simplified for height=1)
    for c_in in range(channels):
        idx_0 = c_in * in_width + src_x0
        idx_1 = c_in * in_width + src_x1
        
        val_0 = tl.load(small_ptr + idx_0, mask=False)
        val_1 = tl.load(small_ptr + idx_1, mask=False)
        
        interpolated = (1 - dx) * val_0 + dx * val_1
    
    # Multiply with large tensor
    large_idx = c * out_height * out_width + h * out_width + w
    large_val = tl.load(large_ptr + large_idx, mask=False)
    
    result = interpolated * large_val
    tl.store(output_ptr + offsets, result, mask=mask)

# Kernel wrapper
@torch.fx.wrap
def fused_interpolate_multiply(small_tensor, large_tensor):
    """Fused bilinear interpolate + multiplication wrapper function"""
    batch_size, channels, in_height, in_width = small_tensor.shape
    _, _, out_height, out_width = large_tensor.shape
    
    # Special case optimization for [1, C, 1, W] -> [1, C, H, W] (common in MobileNetV3)
    if in_height == 1 and batch_size == 1:
        output = torch.empty((batch_size, channels, out_height, out_width), 
                           dtype=small_tensor.dtype, device=small_tensor.device)
        
        total_elements = channels * out_height * out_width
        BLOCK_SIZE = 1024
        num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        optimized_1w_to_hw_interp_mul_kernel[(num_programs,)](
            small_tensor, large_tensor, output,
            channels, in_width, out_height, out_width,
            BLOCK_SIZE
        )
        return output
    
    # General case
    output = torch.empty((batch_size, channels, out_height, out_width), 
                       dtype=small_tensor.dtype, device=small_tensor.device)
    
    total_elements = batch_size * channels * out_height * out_width
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_interp_mul_kernel[(num_programs,)](
        small_tensor, large_tensor, output,
        batch_size, channels, in_height, in_width, out_height, out_width,
        BLOCK_SIZE
    )
    return output

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_interpolate_multiply