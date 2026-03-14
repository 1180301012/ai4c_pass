import torch
import triton
import triton.language as tl

# Pattern matching function - matches the exact matmul operation from model.py
def pattern(in_2, in_3):
    """Match torch.matmul(in_2, in_3) operation with specific tensor shapes"""
    tmp_2 = torch.matmul(in_2, in_3)
    return tmp_2

# Replacement arguments function
def replacement_args(in_2, in_3):
    """Extract arguments needed for the matrix multiplication replacement"""
    return (in_2, in_3)

# Optimized Triton kernel for matrix-vector multiplication with autotuning
# Add autotuning configurations for better performance
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 512}, num_warps=4),
    ],
    key=['n_rows_x', 'n_cols_x', 'n_cols_y'],
)
@triton.jit
def matmul_kernel(
    x_ptr,  # [M, K]
    y_ptr,  # [K, N]  
    out_ptr,  # [M, N]
    n_rows_x: tl.constexpr,
    n_cols_x: tl.constexpr,
    n_cols_y: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Optimized matrix-vector multiplication kernel for shapes [2, 1152] @ [1152, 1] -> [2, 1]
    
    For this small matrix case, we use a direct parallel approach:
    - Each program handles one row of the output
    - Efficient vector memory access patterns
    """
    row_idx = tl.program_id(0)
    
    # Handle only valid rows
    if row_idx >= n_rows_x:
        return
    
    # Initialize accumulator for this row
    acc = 0.0
    
    # Outer loop over columns with optimal block size
    for k in range(0, n_cols_x, BLOCK_SIZE_N):
        # Define column indices for this block
        cols = tl.arange(0, BLOCK_SIZE_N)
        mask = cols < (n_cols_x - k)  # Mask for remaining elements
        
        # Load current row of x with column blocking
        x_ptrs = x_ptr + row_idx * n_cols_x + k + cols
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0)
        
        # Load corresponding element from y (vector)
        y_ptrs = y_ptr + k
        y_val = tl.load(y_ptrs, mask=k < n_cols_x, other=0.0)
        
        # Accumulate dot product
        acc += tl.sum(x_vals * y_val)
    
    # Store result - simplified for vector output
    if n_cols_y > 0:
        tl.store(out_ptr + row_idx * n_cols_y, acc, mask=(row_idx * n_cols_y) < (n_rows_x * n_cols_y))

# Kernel wrapper that matches the pattern behavior
@torch.fx.wrap
def matmul_triton(x, y):
    """
    High-performance matrix multiplication for shapes [M, K] @ [K, N] -> [M, N]
    Optimized for the specific case where N=1 (vector output)
    """
    # Get tensor properties
    n_rows_x = x.shape[0]
    n_cols_x = x.shape[1]
    n_cols_y = y.shape[1]
    
    # Create output tensor
    out = torch.empty((n_rows_x, n_cols_y), dtype=x.dtype, device=x.device)
    
    # Launch kernel - autotuner will automatically pick best config
    n_programs = n_rows_x
    
    matmul_kernel[(n_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_rows_x=n_rows_x,
        n_cols_x=n_cols_x,
        n_cols_y=n_cols_y,
    )
    
    return out

# Replacement function (returns function reference, not a call)
def replacement_func():
    return matmul_triton