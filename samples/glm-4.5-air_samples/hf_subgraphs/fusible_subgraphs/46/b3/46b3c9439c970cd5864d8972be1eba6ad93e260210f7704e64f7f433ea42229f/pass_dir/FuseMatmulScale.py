import torch
import triton
import triton.language as tl

# Pattern matching: matmul + scalar multiplication  
def pattern(x, y, scalar):
    tmp_0 = torch.matmul(x, y)
    tmp_1 = tmp_0 * scalar
    return tmp_1

# Argument extraction for replacement
def replacement_args(x, y, scalar):
    return (x, y, scalar)

# Simple and reliable kernel for matmul + scaling
@triton.jit
def fused_matmul_scale_kernel(
    x_ptr,
    y_ptr,
    scalar_ptr,
    out_ptr,
    n_m,
    n_k,
):
    # Each program handles one row of the output
    m = tl.program_id(0)
    
    # Load scalar value
    scalar = tl.load(scalar_ptr)
    
    # Simple dot product computation
    acc = 0.0
    for k in range(n_k):
        x_val = tl.load(x_ptr + m * n_k + k)
        y_val = tl.load(y_ptr + k)
        acc += x_val * y_val
    
    # Apply scaling
    result = acc * scalar
    tl.store(out_ptr + m, result)

# Wrapper function for fused matmul + scaling
@torch.fx.wrap
def fused_matmul_scale(x, y, scalar):
    # For our specific case: x=[2, 512], y=[512, 1] -> output=[2, 1]
    n_m, n_k = x.shape
    assert y.shape == (n_k, 1), f"Expected y shape {(n_k, 1)}, got {y.shape}"
    
    out = torch.empty((n_m, 1), device=x.device, dtype=x.dtype)
    
    num_programs = n_m
    
    # Launch kernel with appropriate grid
    fused_matmul_scale_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        scalar_ptr=scalar if scalar.dim() > 0 else scalar.unsqueeze(0),
        out_ptr=out,
        n_m=n_m,
        n_k=n_k,
    )
    
    return out

def replacement_func():
    return fused_matmul_scale