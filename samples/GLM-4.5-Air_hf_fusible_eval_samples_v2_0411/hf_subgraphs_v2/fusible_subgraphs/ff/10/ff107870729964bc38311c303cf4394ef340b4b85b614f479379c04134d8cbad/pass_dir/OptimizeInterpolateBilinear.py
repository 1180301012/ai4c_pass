import torch
import triton
import triton.language as tl

@triton.jit
def optimized_bilinear_interpolate_kernel(
    input_ptr,
    output_ptr,
    batch, channels, in_height, in_width,
    out_height, out_width,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of output pixels
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (batch * channels * out_height * out_width)
    
    # Convert linear offset to 4D coordinates
    linear_idx = offsets
    b = linear_idx // (channels * out_height * out_width)
    c = (linear_idx // (out_height * out_width)) % channels
    h = (linear_idx // out_width) % out_height
    w = linear_idx % out_width
    
    # Calculate scaling factors
    scale_y = (in_height - 1) / (out_height - 1) if out_height > 1 else 0
    scale_x = (in_width - 1) / (out_width - 1) if out_width > 1 else 0
    
    # Calculate source coordinates
    src_y = h * scale_y
    src_x = w * scale_x
    
    # Get integer parts and fractional parts
    y0 = tl.cast(tl.floor(src_y), tl.int32)
    y1 = tl.minimum(y0 + 1, in_height - 1)
    x0 = tl.cast(tl.floor(src_x), tl.int32)
    x1 = tl.minimum(x0 + 1, in_width - 1)
    
    # Get fractional parts for interpolation
    dy = src_y - y0
    dx = src_x - x0
    
    # Calculate linear indices for the 4 neighboring pixels
    src_idx_00 = b * channels * in_height * in_width + c * in_height * in_width + y0 * in_width + x0
    src_idx_01 = b * channels * in_height * in_width + c * in_height * in_width + y0 * in_width + x1
    src_idx_10 = b * channels * in_height * in_width + c * in_height * in_width + y1 * in_width + x0
    src_idx_11 = b * channels * in_height * in_width + c * in_height * in_width + y1 * in_width + x1
    
    # Load the 4 neighboring pixels
    I00 = tl.load(input_ptr + src_idx_00, mask=mask, other=0.0)
    I01 = tl.load(input_ptr + src_idx_01, mask=mask, other=0.0)
    I10 = tl.load(input_ptr + src_idx_10, mask=mask, other=0.0)
    I11 = tl.load(input_ptr + src_idx_11, mask=mask, other=0.0)
    
    # Bilinear interpolation
    top = I00 + (I01 - I00) * dx
    bottom = I10 + (I11 - I10) * dx
    output = top + (bottom - top) * dy
    
    # Store result
    output_idx = b * channels * out_height * out_width + c * out_height * out_width + h * out_width + w
    tl.store(output_ptr + output_idx, output, mask=mask)

@torch.fx.wrap
def optimized_bilinear_interpolate(input, size=(512, 512), mode='bilinear', align_corners=False):
    """Optimized bilinear interpolation using Triton"""
    batch, channels, in_height, in_width = input.shape
    out_height, out_width = size
    
    output = torch.empty(batch, channels, out_height, out_width, dtype=input.dtype, device=input.device)
    
    n_elements = output.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    optimized_bilinear_interpolate_kernel[(num_programs,)](
        input_ptr=input,
        output_ptr=output,
        batch=batch, channels=channels, 
        in_height=in_height, in_width=in_width,
        out_height=out_height, out_width=out_width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def pattern(input, size, mode, align_corners):
    """Match any interpolate operation"""
    return torch.nn.functional.interpolate(input, size=size, mode=mode, align_corners=align_corners)

def replacement_args(input, size, mode, align_corners):
    return (input, size, align_corners)

def replacement_func():
    return optimized_bilinear_interpolate