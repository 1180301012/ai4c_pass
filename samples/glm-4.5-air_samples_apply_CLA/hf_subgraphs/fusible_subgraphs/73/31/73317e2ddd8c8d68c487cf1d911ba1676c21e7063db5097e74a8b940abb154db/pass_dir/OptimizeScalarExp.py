import torch
from torch import device
import triton
import triton.language as tl

def pattern(x):
    exp_val = x.exp()
    # Include the redundant device transfer that's in the original computation
    transfer_val = exp_val.to(device=device(type='cuda', index=0))
    return (transfer_val,)

def replacement_args(x):
    return (x,)

@triton.jit
def scalar_exp_kernel(x_ptr, out_ptr):
    # Simplified scalar kernel without block size overhead
    x = tl.load(x_ptr)
    # Use tl.math.exp instead of torch.exp
    out = tl.math.exp(x)
    tl.store(out_ptr, out)

@torch.fx.wrap
def optimized_scalar_exp_kernel(x):
    # Create output tensor directly on correct device to eliminate redundant .to() call
    out = torch.empty_like(x)
    
    # Launch simplified kernel
    scalar_exp_kernel[(1,)](x_ptr=x, out_ptr=out)
    
    return out

def replacement_func():
    return optimized_scalar_exp_kernel