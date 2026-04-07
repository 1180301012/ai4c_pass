import torch
import triton
import triton.language as tl

def pattern(tmp_5):
    # Match just the expand operation
    tmp_6 = tmp_5.expand(3, -1, -1)
    return tmp_6

def replacement_args(tmp_5):
    return (tmp_5,)

@triton.jit
def simple_expand_kernel(
    input_ptr,
    output_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute global offsets
    program_id = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Create mask for columns
    col_mask = col_offsets < n_cols
    
    # Load the input data
    input_data = tl.load(input_ptr + col_offsets, mask=col_mask, other=0)
    
    # Store the expanded version - all 3 copies are the same
    for i in range(3):
        tl.store(output_ptr + i * n_rows * n_cols + col_offsets, 
                input_data, mask=col_mask)

@torch.fx.wrap
def simple_expand_optimized(tmp_5):
    """Optimized version of expand(3, -1, -1)"""
    input_shape = tmp_5.shape
    n_rows = input_shape[0]
    n_cols = input_shape[1]
    
    # Output shape should be (3, n_rows, n_cols)
    output_shape = (3, n_rows, n_cols)
    
    # Create output tensor
    tmp_6 = torch.empty(output_shape, dtype=tmp_5.dtype, device=tmp_5.device)
    
    # For small tensors, use simple CPU-based expansion
    if n_rows * n_cols <= 1024:
        tmp_6 = tmp_5.expand(3, -1, -1)
        return tmp_6
    
    # Set up grid and launch kernel for larger tensors
    BLOCK_SIZE = 1024
    n_programs = 1
    
    simple_expand_kernel[(n_programs,)](
        input_ptr=tmp_5,
        output_ptr=tmp_6,
        n_rows=n_rows,
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return tmp_6

def replacement_func():
    return simple_expand_optimized