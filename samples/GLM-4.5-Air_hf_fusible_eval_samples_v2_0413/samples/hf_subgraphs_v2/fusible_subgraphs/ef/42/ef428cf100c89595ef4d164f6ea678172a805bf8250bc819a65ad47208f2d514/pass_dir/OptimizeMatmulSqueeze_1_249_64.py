import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(x, y):
    """
    Match the computation: matmul + squeeze(1)
    x: [1, 1, 249] - first input tensor
    y: [1, 249, 64] - second input tensor
    Returns: squeezed result [1, 64]
    """
    matmul = torch.matmul(x, y)
    tmp_1 = matmul.squeeze(1)
    return tmp_1

# Argument extraction function
def replacement_args(x, y):
    """Extract arguments needed for the replacement"""
    return (x, y)

# Triton kernel for fused matmul - single program computes entire output
@triton.jit
def matmul_squeeze_kernel_1_249_64_single_program(
    x_ptr,
    y_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
):
    """
    Optimized kernel for matmul + squeeze operation
    Single program computes entire [1, N] output efficiently
    """
    # Since M=1, we only need one program to compute all N elements
    # Each program computes one element of the output
    
    # Program ID determines which output column we compute
    pid = tl.program_id(0)
    
    # If program ID is out of bounds, return
    if pid >= N:
        return
    
    # Compute the single output element: sum over k of x[0, 0, k] * y[0, k, pid]
    acc = 0.0
    
    for k in range(K):
        # Load x[0, 0, k] directly (contiguous access)
        x_val = tl.load(x_ptr + k, other=0.0, mask=k < K).to(tl.float32)
        
        # Load y[0, k, pid] using proper indexing
        y_offset = k * N + pid
        y_val = tl.load(y_ptr + y_offset, other=0.0, mask=(k < K) & (pid < N)).to(tl.float32)
        
        acc += x_val * y_val
    
    # Store the result
    tl.store(out_ptr + pid, acc)

# Alternative kernel with vectorized loads (not used in current implementation)
@triton.jit
def matmul_squeeze_kernel_1_249_64_vec(
    x_ptr,
    y_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
):
    """
    Vectorized version (kept as reference)
    """
    pid = tl.program_id(0)
    
    if pid >= N:
        return
    
    acc = 0.0
    for k in range(K):
        x_val = tl.load(x_ptr + k, mask=k < K, other=0.0).to(tl.float32)
        y_offset = k * N + pid
        y_val = tl.load(y_ptr + y_offset, mask=(k < K) & (pid < N), other=0.0).to(tl.float32)
        acc += x_val * y_val
    
    tl.store(out_ptr + pid, acc)

# Kernel wrapper
@torch.fx.wrap
def fused_matmul_squeeze(x, y):
    """
    Fused matmul and squeeze operation
    Input shapes: x=[1, 1, 249], y=[1, 249, 64]
    Output shape: [1, 64]
    """
    M = 1  # Output first dimension (will be squeezed)
    N = 64  # Output second dimension  
    K = 249  # Contracting dimension
    
    # Grid: one program per output column (N=64 programs total)
    grid = (N,)
    
    # Create output tensor as [N] 
    out = torch.empty((N,), dtype=x.dtype, device=x.device)
    
    # Launch the single-program kernel
    matmul_squeeze_kernel_1_249_64_single_program[grid](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        M=M,
        N=N,
        K=K,
    )
    
    # Reshape to [1, N] to match expected output before squeezing
    out = out.reshape((1, N))
    
    # Since we know M=1, we can automatically squeeze it
    return out.squeeze(0)

# Replacement function
def replacement_func():
    return fused_matmul_squeeze