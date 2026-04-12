import torch
import triton
import triton.language as tl

# Pattern matching function - match just the addition operation
def pattern(x, y):
    """Match: broadcast addition (equivalent to in-place addition)"""
    # Match: in_1 += in_0 (equivalent to in_1 = in_1 + in_0)
    # We use regular addition but it's semantically equivalent
    result = x + y
    return result

# Argument extraction function
def replacement_args(x, y):
    return (x, y)

# Optimized Triton kernel for addition only
@triton.jit
def addition_kernel(
    x_ptr,           # [128, 1] tensor
    y_ptr,           # [1, 128, 19] tensor
    out_ptr,         # [1, 128, 19] output tensor
    n_m: tl.constexpr,  # num_features = 128
    n_t: tl.constexpr,  # num_time = 19
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID for this thread block (1D across all elements)
    pid = tl.program_id(0)
    total_elements = n_m * n_t
    
    # Each thread handles BLOCK_SIZE elements
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Convert 1D offset to 2D coordinates
    m = offsets % n_m
    t = offsets // n_m
    
    # Load y: [1, 128, 19] -> access [0, m, t]
    y_val = tl.load(y_ptr + m * n_t + t, mask=mask, other=0.0)
    
    # Load x with broadcasting: [128, 1] -> [128, 19]
    # Load x[m, 0] and broadcast across t dimension
    x_val = tl.load(x_ptr + m, mask=m < n_m, other=0.0)
    x_bcast = x_val[:, None]  # Broadcast to [128, 19]
    
    # Add x (broadcasted) to y
    result = x_bcast + y_val
    
    # Store result
    tl.store(out_ptr + m * n_t + t, result, mask=mask)

# Kernel wrapper
@torch.fx.wrap  
def optimized_addition(x, y):
    """Optimized addition operation"""
    n_m = x.shape[0]  # 128
    n_t = y.shape[2]  # 19
    
    total_elements = n_m * n_t
    
    # Use block size of 1024 for good GPU occupancy
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor with same shape as y
    output_shape = y.shape  # [1, 128, 19]
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    # Launch kernel with 1D grid
    addition_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_m=n_m,
        n_t=n_t,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function
def replacement_func():
    return optimized_addition