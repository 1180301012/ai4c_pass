import torch
import triton
import triton.language as tl

def pattern(in_1):
    tmp_0 = in_1.norm(p = 2, dim = -1, keepdim = True)
    tmp_1 = in_1 / tmp_0
    return tmp_1

def replacement_args(in_1):
    return (in_1,)

@triton.jit
def fused_norm_kernel(
    input_ptr,
    norm_ptr,
    output_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # This kernel computes both norm and divides in one pass
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE_N)
    
    # Load complete row in one go for maximum efficiency
    mask = col_offsets < n_cols
    row_ptr = input_ptr + row_idx * n_cols
    row_data = tl.load(row_ptr + col_offsets, mask=mask, other=0.0)
    
    # Compute norm more efficiently using vectorized operations
    squared_sum = tl.sum(row_data * row_data)
    norm = tl.sqrt(squared_sum + 1e-8)  # Add epsilon for stability
    
    # Store norm for this row
    if norm_idx := 0:
        tl.store(norm_ptr + row_idx, norm, mask=True)
    
    # Normalize the row using vectorized division
    normalized_row = row_data / norm
    
    # Store result directly
    output_row_ptr = output_ptr + row_idx * n_cols
    tl.store(output_row_ptr + col_offsets, normalized_row, mask=mask)

@triton.jit
def divide_by_norm_kernel(
    input_ptr,
    norm_ptr,
    output_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE_N: tl.constexpr,
):
    # More efficient approach: compute norm with PyTorch, then divide with Triton
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE_N)
    
    # Load row
    mask = col_offsets < n_cols
    row_ptr = input_ptr + row_idx * n_cols
    row_data = tl.load(row_ptr + col_offsets, mask=mask, other=0.0)
    
    # Load pre-computed norm (scalar)
    norm = tl.load(norm_ptr + row_idx)
    
    # Normalize with stable division
    normalized_row = row_data / (norm + 1e-8)
    
    # Store result
    output_row_ptr = output_ptr + row_idx * n_cols
    tl.store(output_row_ptr + col_offsets, normalized_row, mask=mask)

@torch.fx.wrap
def fused_norm_div(x):
    """
    Efficient fusion: compute norm with PyTorch, divide with Triton
    This leverages PyTorch's highly optimized norm while using Triton
    for the vectorized division.
    """
    n_rows, n_cols = x.shape
    
    # Compute norms using PyTorch's highly optimized L2 norm
    # This is much more efficient than computing norm in Triton
    norms = x.norm(p=2, dim=-1, keepdim=True)  # Shape: [n_rows, 1]
    
    # Use Triton for the division operation only for better efficiency
    BLOCK_SIZE = 1024
    num_programs = n_rows
    
    output = torch.empty_like(x)
    
    divide_by_norm_kernel[(num_programs,)](
        input_ptr=x,
        norm_ptr=norms,
        output_ptr=output,
        n_rows=n_rows,
        n_cols=n_cols,
        BLOCK_SIZE_N=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_norm_div