import torch
from torch import device
import triton
import triton.language as tl

# Pattern matching function - match only the matmul operation
def pattern(a, b):
    """
    Match the matmul operation: torch.matmul(a, b)
    This matches the key computation: matmul = torch.matmul(in_2, in_3)
    """
    result = torch.matmul(a, b)
    return result

# Argument extraction function
def replacement_args(a, b):
    return (a, b)

# Simple Triton kernel for matrix multiplication where B has 1 column
@triton.jit
def simple_matmul_kernel(
    a_ptr,           # Pointer to matrix A [M, K]
    b_ptr,           # Pointer to matrix B [K, 1] 
    out_ptr,         # Pointer to output [M, 1]
    M,               # Number of rows in A and output
    K,               # Number of columns in A and rows in B  
    stride_a_m,      # Stride for dimension M in A
    stride_a_k,      # Stride for dimension K in A
    stride_b_k,      # Stride for dimension K in B
    stride_out_m,    # Stride for dimension M in output
):
    # Program ID for M dimension (each program handles one row)
    pid_m = tl.program_id(0)
    
    # Check if this program is within bounds
    if pid_m >= M:
        return
    
    # Each program computes one row of the output
    # Initialize accumulator for this row
    acc = 0.0
    
    # Loop over K dimension (columns of A/rows of B)
    # We'll process one element at a time for simplicity
    for k in range(K):
        # Load one element from the current row of A
        a_offset = pid_m * stride_a_m + k * stride_a_k
        a_val = tl.load(a_ptr + a_offset)
        
        # Load corresponding element from B (single column)
        b_offset = k * stride_b_k
        b_val = tl.load(b_ptr + b_offset)
        
        # Multiply and accumulate
        acc += a_val * b_val
    
    # Store the result for this row
    out_offset = pid_m * stride_out_m
    tl.store(out_ptr + out_offset, acc)

# Kernel wrapper decorated with @torch.fx.wrap as required
@torch.fx.wrap
def optimized_matmul(a, b):
    """
    Optimized matrix multiplication specifically for cases where b has only 1 column.
    This is more efficient than general matrix multiplication for this specific pattern.
    """
    # Get tensor shapes
    M, K = a.shape
    N = b.shape[1]  # Should be 1 for this optimization
    
    assert N == 1, "This optimization is only for matrices where second matrix has 1 column"
    assert a.is_cuda and b.is_cuda, "Both input matrices must be on CUDA"
    
    # Create output tensor - use same dtype as input tensors
    out = torch.empty((M, N), dtype=a.dtype, device=a.device)
    
    # Launch one program per row
    simple_matmul_kernel[M,](
        a_ptr=a,
        b_ptr=b,
        out_ptr=out,
        M=M,
        K=K,
        stride_a_m=a.stride(0),
        stride_a_k=a.stride(1), 
        stride_b_k=b.stride(0),
        stride_out_m=out.stride(0)
    )
    
    return out

# Replacement function (MUST return a function reference, not call it)
def replacement_func():
    return optimized_matmul