import torch
import triton
import triton.language as tl

# Pattern matching function - matches QuickGELU + dropout
def pattern(in_0):
    """
    Match QuickGELU activation: x * sigmoid(1.702 * x) followed by dropout
    """
    tmp_0 = 1.702 * in_0
    tmp_1 = torch.sigmoid(tmp_0)
    tmp_2 = in_0 * tmp_1
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.0, False, False)
    return tmp_3

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Simple 1D QuickGELU kernel
@triton.jit  
def quickgelu_kernel(
    x_ptr,
    out_ptr,
    N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    
    x = tl.load(x_ptr + offs, mask=mask)
    out = x * tl.sigmoid(x * 1.702)
    tl.store(out_ptr + offs, out, mask=mask)

# Kernel wrapper
@torch.fx.wrap
def quickgelu_triton(x):
    out = torch.empty_like(x)
    N = x.numel()
    
    # Try BLOCK=256, num_warps=1
    BLOCK = 256
    grid = ((N + BLOCK - 1) // BLOCK,)
    
    quickgelu_kernel[grid](x, out, N, BLOCK=BLOCK, num_warps=1)
    return out

# Replacement function - returns the kernel wrapper function
def replacement_func():
    return quickgelu_triton