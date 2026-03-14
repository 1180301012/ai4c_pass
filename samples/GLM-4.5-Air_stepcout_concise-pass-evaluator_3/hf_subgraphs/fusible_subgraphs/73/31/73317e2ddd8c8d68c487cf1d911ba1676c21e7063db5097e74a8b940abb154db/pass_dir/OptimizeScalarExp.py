import torch
from torch import device
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0):
    tmp_0 = in_0
    tmp_1 = tmp_0.exp()
    tmp_2 = tmp_1.to(device=device(type='cuda', index=0))
    return (tmp_2,)

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Optimized scalar exponential kernel using direct hardware operations
@triton.jit
def fast_scalar_exp_kernel(
    x_ptr,
    out_ptr,
):
    # Direct hardware-accelerated exponential for best performance
    x = tl.load(x_ptr)
    
    # Use fast math operations for exponential
    out = tl.exp(x)
    
    # Store result
    tl.store(out_ptr, out)

# Optimized scalar exponential wrapper
@torch.fx.wrap  
def optimized_scalar_exp(x):
    # Create output tensor (will be on same device as input)
    out = torch.empty_like(x)
    
    # Launch the fast kernel
    fast_scalar_exp_kernel[(1,)](
        x_ptr=x,
        out_ptr=out,
    )
    
    return out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return optimized_scalar_exp