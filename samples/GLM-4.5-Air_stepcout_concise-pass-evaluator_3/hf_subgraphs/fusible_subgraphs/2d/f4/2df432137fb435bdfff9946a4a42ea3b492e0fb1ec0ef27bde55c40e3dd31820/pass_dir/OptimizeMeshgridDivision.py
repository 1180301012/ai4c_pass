import torch
import triton
import triton.language as tl
from torch import device

# Pattern matching function - matches meshgrid + flattening + division operations
def pattern(in_0, in_1):
    # Match key computation: meshgrid, flattening, and normalization
    tmp_1 = torch.arange(8, dtype=torch.float32, device=device(type='cuda', index=0))
    tmp_2 = torch.functional.meshgrid(in_1, tmp_1)
    tmp_3 = tmp_2[0]
    tmp_4 = tmp_2[1]
    tmp_5 = tmp_3.flatten()
    tmp_6 = tmp_4.flatten()
    tmp_7 = in_0.reshape(1, -1)
    tmp_8 = tmp_5.unsqueeze(-1)
    tmp_9 = tmp_8 / tmp_7
    tmp_10 = tmp_6.unsqueeze(-1)
    tmp_11 = tmp_10 / tmp_7
    return (tmp_9, tmp_11)

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Optimized kernel using Triton
@triton.jit
def meshgrid_division_kernel(
    in_0_ptr,           # Input tensor (positional encoding)
    in_1_ptr,           # Input tensor (arange values)
    out_div_x_ptr,      # Normalized x coordinates
    out_div_y_ptr,      # Normalized y coordinates
    in_0_size,          # Size of in_0 tensor
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of coordinates
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < 8 * in_0_size  # Total number of coordinates
    
    # Calculate x and y indices from flattened offset
    x_idx = offsets % 8  # Fixed 8 from arange
    y_idx = (offsets // 8) % in_0_size
    
    # Load positional encoding values
    norm_factor = tl.load(in_0_ptr + y_idx, mask=mask, other=0.0)
    
    # Normalize coordinates
    x_values = tl.load(in_1_ptr + x_idx, mask=mask, other=0.0) / norm_factor
    y_values = tl.cast(x_idx, tl.float32) / norm_factor
    
    # Store normalized results
    tl.store(out_div_x_ptr + offsets, x_values, mask=mask)
    tl.store(out_div_y_ptr + offsets, y_values, mask=mask)

@torch.fx.wrap
def optimized_meshgrid_division(in_0, in_1):
    """Optimized meshgrid division computation"""
    in_0_size = in_0.shape[0]
    total_coords = 8 * in_0_size
    
    # Create output tensors
    out_div_x = torch.empty(total_coords, dtype=torch.float32, device=in_0.device)
    out_div_y = torch.empty(total_coords, dtype=torch.float32, device=in_0.device)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    num_programs = (total_coords + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    meshgrid_division_kernel[(num_programs,)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_div_x_ptr=out_div_x,
        out_div_y_ptr=out_div_y,
        in_0_size=in_0_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out_div_x, out_div_y

# Replacement function (must return a function reference)
def replacement_func():
    return optimized_meshgrid_division