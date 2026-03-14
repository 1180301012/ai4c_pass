import torch
import triton
import triton.language as tl

# Pattern matching function - matches the entire computation chain
def pattern(in_0):
    # Match the exact computation from model.py
    tmp_0 = in_0
    tmp_1 = tmp_0.ne(1)
    tmp_0 = None
    tmp_2 = tmp_1.int()
    tmp_1 = None
    tmp_3 = torch.cumsum(tmp_2, dim=1)
    tmp_4 = tmp_3.type_as(tmp_2)
    tmp_3 = None
    tmp_5 = tmp_4 + 0
    tmp_4 = None
    tmp_6 = tmp_5 * tmp_2
    tmp_5 = tmp_2 = None
    tmp_7 = tmp_6.long()
    tmp_6 = None
    tmp_8 = tmp_7 + 1
    tmp_7 = None
    return (tmp_8,)

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Autotuned Triton kernel with multiple configuration options
@triton.jit
def autotuned_fused_cumsum_kernel(
    input_ptr,
    output_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
):
    # Each program handles one entire row of the input
    row_idx = tl.program_id(0)
    
    # Calculate global memory position for this row
    row_start = row_idx * n_cols
    
    # Load the entire row with optimized memory access
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < row_start + n_cols
    
    # Load input row with memory coalescing optimization
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0)
    
    # Fast computation in registers - optimized vector operations
    # Create mask (elements != 1)
    mask_vals = (input_vals != 1).to(tl.int32)
    
    # Cumsum operation with optimized parallel execution
    cumsum = tl.cumsum(mask_vals, axis=0)
    
    # Apply mask multiplication efficiently
    result = cumsum * mask_vals
    
    # Final type conversion and addition
    result = result.to(tl.int64) + 1
    
    # Store result with optimized memory access
    tl.store(output_ptr + offsets, result, mask=mask)

# Optimized kernel wrapper
@torch.fx.wrap
def autotuned_cumsum_forward(input_tensor):
    # Get tensor dimensions
    n_rows, n_cols = input_tensor.shape
    
    # Smart block size selection based on tensor characteristics
    if n_cols <= 32:
        BLOCK_SIZE = 32    # Small: optimal for caching
    elif n_cols <= 64:
        BLOCK_SIZE = 64    # Optimal vector width
    elif n_cols <= 128:
        BLOCK_SIZE = 128   # Good balance
    elif n_cols <= 256:
        BLOCK_SIZE = 256   # Throughput oriented
    elif n_cols <= 512:
        BLOCK_SIZE = 512   # Large tensor throughput
    elif n_cols <= 1024:
        BLOCK_SIZE = 1024  # Very large tensor optimization
    else:
        BLOCK_SIZE = 2048  # Maximum throughput
    
    # Ensure BLOCK_SIZE is at least n_cols for row-wise processing
    BLOCK_SIZE = max(BLOCK_SIZE, n_cols)
    
    # Calculate optimal grid size
    grid_size = (n_rows,)
    
    # Create output tensor
    output = torch.empty_like(input_tensor, dtype=torch.int64)
    
    # Launch optimized kernel with adaptive configuration
    num_warps = 8 if n_cols <= 256 else 16  # Adaptive warp count
    num_stages = 3 if n_cols >= 256 else 2  # Adaptive staging
    
    # Fix grid syntax to use proper launch configuration
    grid_size = (n_rows,)
    autotuned_fused_cumsum_kernel[grid_size](
        input_ptr=input_tensor,
        output_ptr=output,
        n_rows=n_rows,
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    
    return output

# Replacement function - returns the autotuned optimized kernel
def replacement_func():
    return autotuned_cumsum_forward