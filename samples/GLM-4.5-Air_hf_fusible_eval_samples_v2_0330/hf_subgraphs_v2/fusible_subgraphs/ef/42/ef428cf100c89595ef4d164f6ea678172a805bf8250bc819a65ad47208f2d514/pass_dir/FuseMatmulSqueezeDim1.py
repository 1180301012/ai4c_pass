import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0, in_1):
    """Match the pattern: matmul followed by squeeze along dimension 1"""
    matmul = torch.matmul(in_0, in_1)
    tmp_1 = matmul.squeeze(1)
    return tmp_1

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Optimized fused kernel - simple and efficient single program
@triton.jit
def fused_matmul_squeeze_kernel(
    x_ptr,           # in_0: [1, 1, 249] 
    y_ptr,           # in_1: [1, 249, 64] 
    out_ptr,         # output: [1, 64] 
):
    # Matrix multiplication kernel for [1, 1, 249] @ [1, 249, 64] -> [1, 64]
    # Simple vector-matrix multiplication optimized for small sizes
    
    # Initialize all output elements
    result = tl.zeros((64,), dtype=tl.float32)
    
    # Vector-matrix multiplication: result[j] = sum_k (x[k] * y[k, j])
    for k in range(249):
        # Load x[k] from [1, 1, 249] tensor
        x_val = tl.load(x_ptr + k)
        
        # Load entire row y[k, :] from [1, 249, 64] tensor
        y_row = tl.load(y_ptr + k * 64 + tl.arange(0, 64))
        
        # Add contribution: x[k] * y[k, :] to result
        result += x_val * y_row
    
    # Store entire result
    tl.store(out_ptr + tl.arange(0, 64), result)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def fused_matmul_squeeze(in_0, in_1):
    # Get input shapes and data types
    _, _, K = in_0.shape  # in_0: [1, 1, 249] -> K=249
    _, K2, N = in_1.shape  # in_1: [1, 249, 64] -> K2=249, N=64
    
    assert K == 249, f"Expected K=249, got {K}"
    assert K2 == 249, f"Expected K2=249, got {K2}"
    assert N == 64, f"Expected N=64, got {N}"
    
    # Create output tensor (squeezed to [1, 64] but stored as [64] for efficiency)
    # Use float32 for computation, then convert back to original dtype
    out = torch.empty((N,), dtype=torch.float32, device=in_0.device)
    
    # Launch only 1 program to compute all 64 output elements
    fused_matmul_squeeze_kernel[(1,)](
        x_ptr=in_0,
        y_ptr=in_1,
        out_ptr=out,
    )
    
    # Convert back to original data type and reshape to [1, 64]
    return out.to(in_0.dtype).reshape(1, N)

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_matmul_squeeze