import torch
import triton
import triton.language as tl

# Pattern matching function - matches the view-multiply pattern
def pattern(in_1, in_2):
    """Match: in_1.view(-1, 1) * in_2"""
    tmp_0 = in_1.view(-1, 1)
    tmp_1 = tmp_0 * in_2
    return tmp_1

# Argument extraction function  
def replacement_args(in_1, in_2):
    return (in_1, in_2)

# Optimized Triton kernel for broadcasted multiplication
@triton.jit
def broadcast_multiply_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel for column vector tensor multiplication"""
    # Each program handles one row
    row_idx = tl.program_id(0)
    
    # Get pointer offsets for this row
    a_offset = row_idx  # a is [n_rows], accessed as [row_idx]
    b_offset = row_idx * n_cols  # b is [n_rows, n_cols], start of row
    
    # Load scalar from column vector a and row vector b
    a_val = tl.load(a_ptr + a_offset)
    b_row = tl.load(b_ptr + b_offset + tl.arange(0, BLOCK_SIZE))
    
    # Broadcast scalar to full row and multiply
    result = a_val * b_row
    
    # Store result
    out_offset = row_idx * n_cols
    tl.store(out_ptr + out_offset + tl.arange(0, BLOCK_SIZE), result, mask=tl.arange(0, BLOCK_SIZE) < n_cols)

# Kernel wrapper that matches original interface
@torch.fx.wrap
def fused_multiply_broadcast(a, b):
    """Fused operation: view(-1, 1) * broadcast multiplication"""
    n_rows = a.shape[0]
    n_cols = b.shape[1]
    
    # Get total number of elements
    n_elements = n_rows * n_cols
    
    # Optimized block size based on tensor widths
    BLOCK_SIZE = 128  # Good for GPU warps
    num_rows = n_rows
    
    # Allocate output tensor with same dtype as b
    out = torch.empty((n_rows, n_cols), dtype=b.dtype, device=b.device)
    
    # Launch Triton kernel
    broadcast_multiply_kernel[(num_rows,)](
        a_ptr=a,
        b_ptr=b,
        out_ptr=out,
        n_rows=n_rows,
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function - returns the optimized kernel wrapper
def replacement_func():
    return fused_multiply_broadcast