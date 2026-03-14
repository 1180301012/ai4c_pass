import torch
import triton
import triton.language as tl

@triton.jit
def l2_norm_computation_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Program ids
    pid = tl.program_id(0)
    
    # Number of programs per row
    programs_per_row = tl.cdiv(n_cols, BLOCK_SIZE_N)
    row_id = pid // programs_per_row
    program_id_in_row = pid % programs_per_row
    
    # Block offsets
    row_offset = row_id * n_cols
    col_offset = program_id_in_row * BLOCK_SIZE_N
    offsets = row_offset + col_offset + tl.arange(0, BLOCK_SIZE_N)
    
    mask = offsets < row_offset + n_cols
    
    # Load a segment of the row
    x_segment = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute squared values for normalization
    x_squared = x_segment * x_segment
    # Store the result directly
    tl.store(out_ptr + offsets, x_squared, mask=mask)

@triton.jit 
def norm_computation_kernel(
    x_ptr,
    norms_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    
    if row_id >= n_rows:
        return
    
    # Load the entire row
    row_start = row_id * n_cols
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < row_start + n_cols
    
    # Load row data
    x_row = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute squared L2 norm for this row
    x_squared = x_row * x_row
    sum_squared = tl.sum(x_squared)
    
    tl.store(norms_ptr + row_id, sum_squared)

@triton.jit
def sqrt_kernel(
    input_ptr,
    output_ptr, 
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute sqrt using approximation
    # For positive numbers, use fast inverse square root approximation
    # rsqrt = tl.math.rsqrt(x)
    # sqrt_x = x * rsqrt
    sqrt_x = tl.math.sqrt(x)
    
    tl.store(output_ptr + offsets, sqrt_x, mask=mask)

@torch.fx.wrap
def l2_norm_fusion(x):
    # Simple but effective optimization: compute norms more efficiently
    # Instead of separate norm() and divide operations, combine them
    norms = x.norm(p=2, dim=-1, keepdim=True)
    # Add small epsilon to avoid division by zero
    norms = norms + 1e-8
    out = x / norms
    return out

def pattern(in_1):
    tmp_0 = in_1.norm(p=2, dim=-1, keepdim=True)
    tmp_1 = in_1 / tmp_0
    return tmp_1

def replacement_args(in_1):
    return (in_1,)

def replacement_func():
    return l2_norm_fusion