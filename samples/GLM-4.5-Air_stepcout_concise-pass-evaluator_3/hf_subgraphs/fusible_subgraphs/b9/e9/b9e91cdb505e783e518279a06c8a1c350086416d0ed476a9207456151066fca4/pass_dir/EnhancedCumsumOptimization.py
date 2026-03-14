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

# Enhanced Optimal Triton kernel with better memory optimization
@triton.jit
def enhanced_fused_cumsum_kernel(
    input_ptr,
    output_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one entire row of the input
    row_idx = tl.program_id(0)
    
    # Calculate global memory position for this row
    row_start = row_idx * n_cols
    
    # Load the entire row with optimal vectorization
    # Use BLOCK_SIZE that matches hardware vector width
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < row_start + n_cols
    
    # Load input row - this is memory-bound, so optimize carefully
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0)
    
    # Fast path: compute all operations in registers
    # This eliminates intermediate memory operations
    
    # Step 1: Create mask (elements != 1)
    mask_vals = (input_vals != 1).to(tl.int32)
    
    # Step 2: Compute cumsum with optimized vector ops
    cumsum = tl.cumsum(mask_vals, axis=0)
    
    # Step 3: Apply mask multiplication (zeros out positions where original was 1)
    masked_cumsum = cumsum * mask_vals
    
    # Step 4: Convert to long and add 1 in one operation
    result = masked_cumsum.to(tl.int64) + 1
    
    # Store result directly - avoid intermediate memory writes
    tl.store(output_ptr + offsets, result, mask=mask)

# Performance-optimized kernel wrapper
@torch.fx.wrap
def enhanced_cumsum_forward(input_tensor):
    # Get tensor dimensions
    n_rows, n_cols = input_tensor.shape
    
    # Optimized block size selection based on hardware capabilities
    if n_cols <= 32:
        BLOCK_SIZE = 32  # Small tensor: use smallest efficient size
    elif n_cols <= 64:
        BLOCK_SIZE = 64  # Small-medium: optimal cache usage
    elif n_cols <= 128:
        BLOCK_SIZE = 128  # Medium: good vectorization
    elif n_cols <= 256:
        BLOCK_SIZE = 256  # Large to medium: balance memory vs compute
    elif n_cols <= 512:
        BLOCK_SIZE = 512  # Large: good memory throughput
    elif n_cols <= 1024:
        BLOCK_SIZE = 1024  # Very large: maximize throughput
    else:
        BLOCK_SIZE = 2048  # Extra large: maximize memory access
    
    # Ensure BLOCK_SIZE is at least n_cols to process each row in one program
    BLOCK_SIZE = max(BLOCK_SIZE, n_cols)
    
    # Calculate optimal grid size
    grid_size = (n_rows,)
    
    # Create output tensor with proper dtype and shape
    output = torch.empty_like(input_tensor, dtype=torch.int64)
    
    # Launch enhanced kernel
    enhanced_fused_cumsum_kernel[grid_size](
        input_ptr=input_tensor,
        output_ptr=output,
        n_rows=n_rows,
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Replacement function - returns the enhanced optimized kernel
def replacement_func():
    return enhanced_cumsum_forward