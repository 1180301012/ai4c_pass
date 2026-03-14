import torch
import triton
import triton.language as tl
from torch import device

# Pattern matching function - matches the entire positional encoding computation
def pattern(in_0, in_1):
    # Match the entire computation from model.py
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
    tmp_12 = tmp_9.cos()
    tmp_13 = tmp_9.sin()
    return (tmp_12, tmp_11, tmp_13)

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Optimized kernel using Triton
@triton.jit
def positional_encoding_kernel(
    in_0_ptr,           # Input tensor (positional encoding)
    out_cos_ptr,        # Cosine output
    out_sin_ptr,        # Sine output  
    out_normalized_y_ptr,  # Normalized y coordinates
    in_0_size,          # Size of in_0 tensor
    n_coords,           # Number of coordinates (8 * len(in_1))
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of coordinates
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_coords
    
    # Load positional encoding values (we'll use them in a broadcast manner)
    # For each coordinate, we need the corresponding in_0 value based on x position
    x_coord = offsets % len(in_1)  # x coordinate from meshgrid
    in_0_idx = tl.load(in_0_ptr + x_coord, mask=mask, other=0.0)
    
    # Normalize coordinates
    # X coordinates are based on the pattern from in_1
    x_coords_norm = tl.cast(offsets % len(in_1), tl.float32) / in_0_size
    y_coords_norm = tl.cast(offsets // len(in_1), tl.float32) / in_0_size
    
    # Compute trigonometric functions
    cos_vals = tl.cos(x_coords_norm)
    sin_vals = tl.sin(x_coords_norm)
    
    # Store results
    tl.store(out_cos_ptr + offsets, cos_vals, mask=mask)
    tl.store(out_sin_ptr + offsets, sin_vals, mask=mask)
    tl.store(out_normalized_y_ptr + offsets, y_coords_norm, mask=mask)

@torch.fx.wrap
def optimized_positional_encoding(in_0, in_1):
    """Optimized implementation of positional encoding computation"""
    # Get input sizes
    in_0_size = in_0.shape[0]
    n_coords = in_1.shape[0] * in_0_size  # meshgrid size
    
    # Create output tensors
    out_cos = torch.empty(n_coords, dtype=torch.float32, device=in_0.device)
    out_sin = torch.empty(n_coords, dtype=torch.float32, device=in_0.device)
    out_normalized_y = torch.empty(n_coords, dtype=torch.float32, device=in_0.device)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    num_programs = (n_coords + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    positional_encoding_kernel[(num_programs,)](
        in_0_ptr=in_0,
        out_cos_ptr=out_cos,
        out_sin_ptr=out_sin,
        out_normalized_y_ptr=out_normalized_y,
        in_0_size=in_0_size,
        n_coords=n_coords,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out_cos, out_normalized_y, out_sin

# Replacement function (must return a function reference)
def replacement_func():
    return optimized_positional_encoding