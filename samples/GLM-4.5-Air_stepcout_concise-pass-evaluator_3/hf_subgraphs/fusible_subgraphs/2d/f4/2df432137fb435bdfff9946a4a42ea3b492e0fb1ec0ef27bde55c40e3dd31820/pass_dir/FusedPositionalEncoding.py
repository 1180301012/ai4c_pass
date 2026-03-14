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

# Optimized fused kernel for the entire positional encoding computation
@triton.jit
def fused_positional_encoding_kernel(
    in_0_ptr,           # Input tensor (positional encoding)
    in_1_ptr,           # Input tensor (arange values)
    cos_out_ptr,        # Cosine output
    sin_out_ptr,        # Sine output
    normalized_y_ptr,   # Normalized y coordinates output
    in_0_size,          # Size of in_0 tensor
    in_1_size,          # Size of in_1 tensor (should be 8)
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of flattened coordinates
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < in_0_size * in_1_size  # Total number of coordinates
    
    # Calculate x and y indices from flattened offset
    x_coord = offsets % in_1_size  # x coordinate from in_1
    y_coord = offsets // in_1_size  # y coordinate
    
    # Load input values and compute normalization
    in_0_val = tl.load(in_0_ptr + y_coord, mask=mask, other=0.0)
    in_1_val = tl.load(in_1_ptr + x_coord, mask=mask, other=0.0)
    
    # Compute coordinates and normalize
    x_norm = in_1_val / in_0_val
    y_norm = tl.cast(x_coord, tl.float32) / in_0_val
    
    # Compute trigonometric functions on normalized x coordinate
    cos_x = tl.cos(x_norm)
    sin_x = tl.sin(x_norm)
    
    # Store all three results
    tl.store(cos_out_ptr + offsets, cos_x, mask=mask)
    tl.store(normalized_y_ptr + offsets, y_norm, mask=mask)
    tl.store(sin_out_ptr + offsets, sin_x, mask=mask)

@torch.fx.wrap
def optimized_fused_positional_encoding(in_0, in_1):
    """Fused computation of entire positional encoding"""
    in_0_size = in_0.shape[0]
    in_1_size = in_1.shape[0]
    total_coords = in_0_size * in_1_size
    
    # Create output tensors
    cos_out = torch.empty(total_coords, dtype=torch.float32, device=in_0.device)
    sin_out = torch.empty(total_coords, dtype=torch.float32, device=in_0.device)
    normalized_y_out = torch.empty(total_coords, dtype=torch.float32, device=in_0.device)
    
    # Launch kernel with autotuning capability
    BLOCK_SIZE = 1024
    num_programs = (total_coords + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_positional_encoding_kernel[(num_programs,)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        cos_out_ptr=cos_out,
        sin_out_ptr=sin_out,
        normalized_y_ptr=normalized_y_out,
        in_0_size=in_0_size,
        in_1_size=in_1_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return cos_out, normalized_y_out, sin_out

# Replacement function (must return a function reference)
def replacement_func():
    return optimized_fused_positional_encoding