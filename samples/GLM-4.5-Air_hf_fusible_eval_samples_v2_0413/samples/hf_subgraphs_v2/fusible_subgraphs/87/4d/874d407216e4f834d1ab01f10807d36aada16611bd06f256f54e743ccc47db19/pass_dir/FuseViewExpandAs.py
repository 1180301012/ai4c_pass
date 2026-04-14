import torch
import triton
import triton.language as tl

# Pattern matching function - matches the view-expand pattern
def pattern(in_0, target_shape):
    """Match: in_0.view((-1, 1)).expand_as(target_shape)"""
    tmp_2 = in_0.view((-1, 1))
    tmp_3 = tmp_2.expand_as(target_shape)
    return tmp_3

# Argument extraction function
def replacement_args(in_0, tmp_1):
    """tmp_1 provides the target shape for expand_as"""
    target_shape = tmp_1.shape
    return (in_0, target_shape)

# Optimized Triton kernel for efficient column vector expansion
@triton.jit
def expand_column_vector_kernel(
    in_ptr,
    out_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel for expanding column vector to 2D tensor"""
    # Each program handles one row
    row_idx = tl.program_id(0)
    
    # Load scalar from column vector
    val = tl.load(in_ptr + row_idx)
    
    # Create offsets for this row in output tensor
    out_offsets = row_idx * n_cols + tl.arange(0, BLOCK_SIZE)
    
    # Broadcast scalar across the entire row
    result = tl.full((BLOCK_SIZE,), val, dtype=tl.float32)
    
    # Store result with proper masking
    tl.store(out_ptr + out_offsets, result, mask=tl.arange(0, BLOCK_SIZE) < n_cols)

# Kernel wrapper that handles different data types
@torch.fx.wrap
def fused_expand_column_vector(in_tensor, target_shape):
    """Fused operation: view(-1, 1).expand_as(target_tensor)"""
    n_rows = in_tensor.shape[0]
    n_cols = target_shape[1]  # Use target's column dimension
    
    # Determine dtype and device from input
    dtype = in_tensor.dtype
    device = in_tensor.device
    
    # Optimized block size
    BLOCK_SIZE = 128
    num_rows = n_rows
    
    # Allocate output tensor
    out = torch.empty(target_shape, dtype=dtype, device=device)
    
    # Launch Triton kernel
    expand_column_vector_kernel[(num_rows,)](
        in_ptr=in_tensor,
        out_ptr=out,
        n_rows=n_rows,
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function - returns the optimized kernel wrapper
def replacement_func():
    return fused_expand_column_vector