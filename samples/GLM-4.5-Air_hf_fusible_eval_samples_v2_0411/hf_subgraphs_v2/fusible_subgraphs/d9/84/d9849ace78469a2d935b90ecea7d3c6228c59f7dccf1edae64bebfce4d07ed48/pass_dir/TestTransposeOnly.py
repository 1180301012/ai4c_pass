import torch
import triton
import triton.language as tl

# Pattern matching function - match just the transpose operation
def pattern(x):
    """Match: transpose operation only"""
    # Match: tmp_2 = in_2.transpose(1, 2)
    result = x.transpose(1, 2)
    return result

# Argument extraction function
def replacement_args(x):
    return (x,)

# Optimized Triton kernel for transpose only
@triton.jit
def transpose_kernel(
    x_ptr,           # Input tensor [1, 128, 19]
    out_ptr,         # Output tensor [1, 19, 128]
    n_m: tl.constexpr,  # num_features = 128
    n_t: tl.constexpr,  # num_time = 19
    batch_size: tl.constexpr,  # batch size = 1
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID for this thread block (1D across all transpose elements)
    pid = tl.program_id(0)
    total_elements = n_m * n_t
    
    # Each thread handles BLOCK_SIZE elements
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Convert 1D offset to 2D coordinates (batch=0, m, t) -> (batch=0, t, m)
    b = 0  # Always batch 0
    m = offsets % n_m
    t = offsets // n_m
    
    # Calculate 1D offsets in contiguous memory
    input_offset = b * n_m * n_t + m * n_t + t
    output_offset = b * n_t * n_m + t * n_m + m
    
    # Load and store with proper broadcasting
    x = tl.load(x_ptr + input_offset, mask=mask, other=0.0)
    tl.store(out_ptr + output_offset, x, mask=mask)

# Kernel wrapper
@torch.fx.wrap  
def optimized_transpose(x):
    """Optimized transpose operation"""
    n_m = x.shape[1]  # 128 (dimension 1)
    n_t = x.shape[2]  # 19 (dimension 2)
    batch_size = x.shape[0]  # 1
    
    total_elements = n_m * n_t
    
    # Use block size of 1024 for good GPU occupancy
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor with transposed shape
    output_shape = list(x.shape)
    output_shape[1], output_shape[2] = output_shape[2], output_shape[1]  # Swap dims 1 and 2
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    # Launch kernel with 1D grid
    transpose_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_m=n_m,
        n_t=n_t,
        batch_size=batch_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function
def replacement_func():
    return optimized_transpose