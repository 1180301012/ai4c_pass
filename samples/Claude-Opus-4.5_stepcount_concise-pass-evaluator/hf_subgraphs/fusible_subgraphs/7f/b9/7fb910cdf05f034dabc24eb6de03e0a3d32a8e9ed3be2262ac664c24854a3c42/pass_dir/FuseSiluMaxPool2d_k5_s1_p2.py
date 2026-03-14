import torch
import triton
import triton.language as tl

# Pattern matching function - must match the model.py exactly
def pattern(in_0):
    tmp_0 = torch.nn.functional.silu(in_0, inplace=True)
    tmp_1 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    return (tmp_1, tmp_0)

# Extract arguments needed for replacement
def replacement_args(in_0):
    return (in_0,)

# Triton kernel for SiLU activation
@triton.jit
def silu_kernel(
    x_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    x = tl.load(x_ptr + offset, mask=mask)
    # silu(x) = x * sigmoid(x)
    out = x * tl.sigmoid(x)
    tl.store(out_ptr + offset, out, mask=mask)

# Triton kernel for 5x5 MaxPool2D with stride=1, padding=2
@triton.jit
def maxpool2d_5x5_kernel(
    input_ptr, output_ptr,
    H, W, HW, CHW,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Only process valid indices
    valid_mask = offsets < n_elements
    
    # Compute (n, c, h, w) from linear index (NCHW layout)
    w_idx = offsets % W
    h_idx = (offsets // W) % H  
    nc_idx = offsets // HW  # combined n*C + c index
    
    # Base offset for this (n, c) slice
    base_offset = nc_idx * HW
    
    # Initialize max values to -inf
    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)
    
    # Iterate over 5x5 window (padding=2, so window is centered)
    for dy in tl.static_range(-2, 3):
        for dx in tl.static_range(-2, 3):
            h_src = h_idx + dy
            w_src = w_idx + dx
            
            # Check bounds (padding handles out-of-bounds as -inf for max)
            in_bounds = (h_src >= 0) & (h_src < H) & (w_src >= 0) & (w_src < W)
            
            # Compute source index
            src_idx = base_offset + h_src * W + w_src
            
            # Load value (use -inf for out of bounds)
            val = tl.load(input_ptr + src_idx, mask=valid_mask & in_bounds, other=float('-inf'))
            
            # Update max
            max_val = tl.maximum(max_val, val)
    
    # Store result
    tl.store(output_ptr + offsets, max_val, mask=valid_mask)

# Wrapper function that launches the kernels
@torch.fx.wrap
def fused_silu_maxpool(in_0):
    N, C, H, W = in_0.shape
    n_elements = in_0.numel()
    HW = H * W
    CHW = C * H * W
    
    # Allocate output for silu
    silu_out = torch.empty_like(in_0)
    
    # Compute silu
    BLOCK_SIZE = 1024
    grid_silu = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    silu_kernel[grid_silu](in_0, silu_out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Allocate output for max_pool
    maxpool_out = torch.empty_like(in_0)
    
    # Compute max_pool
    grid_maxpool = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    maxpool2d_5x5_kernel[grid_maxpool](
        silu_out, maxpool_out,
        H, W, HW, CHW,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return (maxpool_out, silu_out)

# Replacement function - returns the wrapper function (not a call)
def replacement_func():
    return fused_silu_maxpool