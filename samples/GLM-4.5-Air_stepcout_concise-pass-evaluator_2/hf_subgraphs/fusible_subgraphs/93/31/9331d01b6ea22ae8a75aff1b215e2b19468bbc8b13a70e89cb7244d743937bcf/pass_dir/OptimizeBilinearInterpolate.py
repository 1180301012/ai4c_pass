import torch
import triton
import triton.language as tl

def pattern(x):
    # Matches the bilinear interpolate operation
    tmp = torch.nn.functional.interpolate(x, size=(128, 128), mode='bilinear', align_corners=False)
    return tmp

def replacement_args(x):
    return (x,)

@triton.jit
def bilinear_interpolate_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    channels,
    in_height,
    in_width,
    out_height,
    out_width,
    block_size: tl.constexpr
):
    """Bilinear interpolation kernel using Triton"""
    pid = tl.program_id(0)
    grid_size = batch_size * channels * out_height * out_width
    
    if pid >= grid_size:
        return
    
    # Convert 1D program ID to 4D coordinates
    bid = pid // (channels * out_height * out_width)
    cid = (pid % (channels * out_height * out_width)) // (out_height * out_width)
    h = (pid % (out_height * out_width)) // out_width
    w = pid % out_width
    
    # Calculate sampling coordinates in input space
    x_scale = (in_width - 1) / (out_width - 1) if out_width > 1 else 1.0
    y_scale = (in_height - 1) / (out_height - 1) if out_height > 1 else 1.0
    
    x_src = w * x_scale
    y_src = h * y_scale
    
    # Get four nearest neighbors
    x0 = int(x_src)
    y0 = int(y_src)
    x1 = min(x0 + 1, in_width - 1)
    y1 = min(y0 + 1, in_height - 1)
    
    # Calculate fractional parts for interpolation
    if in_width > 1 and out_width > 1:
        fx = x_src - x0
        fy = y_src - y0
    else:
        fx = fy = 0.0
    
    # Calculate input offsets
    input_offset = bid * channels * in_height * in_width + cid * in_height * in_width
    output_offset = bid * channels * out_height * out_width + cid * out_height * out_width + h * out_width + w
    
    # Load four corner values
    q00 = tl.load(input_ptr + input_offset + y0 * in_width + x0, mask=True, other=0.0)
    q01 = tl.load(input_ptr + input_offset + y0 * in_width + x1, mask=True, other=0.0)
    q10 = tl.load(input_ptr + input_offset + y1 * in_width + x0, mask=True, other=0.0)
    q11 = tl.load(input_ptr + input_offset + y1 * in_width + x1, mask=True, other=0.0)
    
    # Bilinear interpolation
    top = q00 * (1 - fx) + q01 * fx
    bottom = q10 * (1 - fx) + q11 * fx
    result = top * (1 - fy) + bottom * fy
    
    # Store result
    tl.store(output_ptr + output_offset, result)

@triton.jit
def bilinear_interpolate_batch_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    channels,
    in_height,
    in_width,
    out_height,
    out_width,
    block_size: tl.constexpr
):
    """Optimized bilinear interpolation kernel with better memory coalescing"""
    pid = tl.program_id(0)
    num_programs = tl.cdiv(batch_size * channels * out_height * out_width, block_size)
    
    if pid >= num_programs:
        return
    
    # Calculate global offset and iterate through block
    start_offset = pid * block_size
    end_offset = min((pid + 1) * block_size, batch_size * channels * out_height * out_width)
    
    for offset in range(start_offset, end_offset):
        # Convert 1D offset to 4D coordinates
        bid = offset // (channels * out_height * out_width)
        cid = (offset % (channels * out_height * out_width)) // (out_height * out_width)
        h = (offset % (out_height * out_width)) // out_width
        w = offset % out_width
        
        # Calculate sampling coordinates in input space
        x_scale = (in_width - 1) / (out_width - 1) if out_width > 1 else 1.0
        y_scale = (in_height - 1) / (out_height - 1) if out_height > 1 else 1.0
        
        x_src = w * x_scale
        y_src = h * y_scale
        
        # Get four nearest neighbors
        x0 = int(x_src)
        y0 = int(y_src)
        x1 = min(x0 + 1, in_width - 1)
        y1 = min(y0 + 1, in_height - 1)
        
        # Calculate fractional parts for interpolation
        if in_width > 1 and out_width > 1:
            fx = x_src - x0
            fy = y_src - y0
        else:
            fx = fy = 0.0
        
        # Calculate input offsets
        input_offset = bid * channels * in_height * in_width + cid * in_height * in_width + y0 * in_width
        output_offset = bid * channels * out_height * out_width + cid * out_height * out_width + h * out_width + w
        
        # Load four corner values with optimized memory access
        q00 = tl.load(input_ptr + input_offset + x0, mask=True, other=0.0)
        q01 = tl.load(input_ptr + input_offset + x1, mask=True, other=0.0)
        q10 = tl.load(input_ptr + input_offset + in_width + x0, mask=True, other=0.0)
        q11 = tl.load(input_ptr + input_offset + in_width + x1, mask=True, other=0.0)
        
        # Bilinear interpolation
        top = q00 * (1 - fx) + q01 * fx
        bottom = q10 * (1 - fx) + q11 * fx
        result = top * (1 - fy) + bottom * fy
        
        # Store result
        tl.store(output_ptr + output_offset, result)

@torch.fx.wrap
def optimized_bilinear_interpolate(x):
    """Optimized bilinear interpolation function"""
    batch_size, channels, in_height, in_width = x.shape
    out_height, out_width = 128, 128
    
    output = torch.empty((batch_size, channels, out_height, out_width), 
                        dtype=x.dtype, device=x.device)
    
    # Choose kernel configuration based on input size
    if batch_size * channels > 32:  # Use vectorized kernel for larger batches
        grid = triton.cdiv(batch_size * channels * out_height * out_width, 512)
        bilinear_interpolate_batch_kernel[grid](
            x, output,
            batch_size, channels, in_height, in_width,
            out_height, out_width,
            512
        )
    else:  # Use simpler kernel for smaller batches
        grid = (batch_size, channels, out_height, out_width)
        bilinear_interpolate_kernel[grid](
            x, output,
            batch_size, channels, in_height, in_width,
            out_height, out_width,
            32
        )
    
    return output

def replacement_func():
    return optimized_bilinear_interpolate