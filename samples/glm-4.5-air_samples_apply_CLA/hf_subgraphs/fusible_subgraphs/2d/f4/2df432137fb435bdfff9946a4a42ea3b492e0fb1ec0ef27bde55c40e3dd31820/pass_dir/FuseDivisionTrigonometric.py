import torch
import triton
import triton.language as tl
import math

# Pattern matching function for broadcast division + trigonometric operations
def pattern(coord_x, coord_y, scale_tensor):
    # Match the division and trigonometric operations
    tmp_7 = scale_tensor.reshape(1, -1)
    tmp_8 = coord_x.unsqueeze(-1)
    tmp_9 = tmp_8 / tmp_7
    tmp_10 = coord_y.unsqueeze(-1)
    tmp_11 = tmp_10 / tmp_7
    tmp_12 = tmp_9.cos()
    tmp_13 = tmp_9.sin()
    return tmp_12, tmp_11, tmp_13

# Argument extraction function
def replacement_args(coord_x, coord_y, scale_tensor):
    return (coord_x, coord_y, scale_tensor)

# Optimized kernel for fused broadcast division and trigonometric operations
@triton.jit
def fused_div_trig_kernel(
    coord_x_ptr,        # X coordinate tensor pointer [64]
    coord_y_ptr,        # Y coordinate tensor pointer [64]
    scale_ptr,          # Scale tensor pointer [64]
    cos_out_ptr,        # Cosine output pointer [64, 64]
    sin_out_ptr,        # Sine output pointer [64, 64]
    div_out_ptr,        # Division output pointer [64, 64]
    coord_size,         # Size of coordinate tensors (64)
    total_size,         # Total output size (64*64)
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_size
    
    # Calculate 2D coordinates from linear index
    i = offsets // coord_size  # Row index (0-63)
    j = offsets % coord_size   # Column index (0-63)
    
    # Load coordinate and scale values
    coord_x_val = tl.load(coord_x_ptr + j, mask=mask, other=0.0)
    coord_y_val = tl.load(coord_y_ptr + j, mask=mask, other=0.0)
    scale_val = tl.load(scale_ptr + j, mask=mask, other=0.0)
    
    # Compute broadcast division: coord_x_val / scale_val, coord_y_val / scale_val
    # Since we're doing [64,1] / [1,64] broadcasting, each coordinate is divided by all scales
    div_cos_val = coord_x_val / scale_val  # [64, 64] through broadcasting
    div_sin_val = coord_y_val / scale_val  # [64, 64] through broadcasting
    
    # Compute trigonometric functions
    cos_val = tl.cos(div_cos_val)
    sin_val = tl.sin(div_cos_val)
    
    # Store results
    tl.store(cos_out_ptr + offsets, cos_val, mask=mask)
    tl.store(sin_out_ptr + offsets, sin_val, mask=mask)
    tl.store(div_out_ptr + offsets, div_sin_val, mask=mask)

# Kernel wrapper
@torch.fx.wrap
def optimized_fused_div_trig(coord_x, coord_y, scale_tensor):
    # Get tensor sizes
    coord_size = coord_x.size(0)  # Should be 64
    total_size = coord_size * coord_size  # 64*64 = 4096
    
    # Create output tensors
    cos_out = torch.empty(total_size, dtype=coord_x.dtype, device=coord_x.device)
    sin_out = torch.empty(total_size, dtype=coord_x.dtype, device=coord_x.device)
    div_out = torch.empty(total_size, dtype=coord_x.dtype, device=coord_x.device)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    num_programs = (total_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_div_trig_kernel[(num_programs,)](
        coord_x_ptr=coord_x,
        coord_y_ptr=coord_y,
        scale_ptr=scale_tensor,
        cos_out_ptr=cos_out,
        sin_out_ptr=sin_out,
        div_out_ptr=div_out,
        coord_size=coord_size,
        total_size=total_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape outputs to [64, 64]
    cos_reshaped = cos_out.reshape(coord_size, coord_size)
    sin_reshaped = sin_out.reshape(coord_size, coord_size)
    div_reshaped = div_out.reshape(coord_size, coord_size)
    
    return cos_reshaped, div_reshaped, sin_reshaped

# Replacement function
def replacement_func():
    return optimized_fused_div_trig