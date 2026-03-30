import torch
import triton
import triton.language as tl

# Pattern matching function - exactly matches the computation in model.py
def pattern(in_0):
    tmp_0 = torch.nn.functional.gelu(in_0)
    tmp_1 = tmp_0.mean((2, 3), keepdim=True)
    return (tmp_0, tmp_1)

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Simple Triton kernel for GELU activation
@triton.jit
def gelu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    gelu_val = x * 0.5 * (1.0 + tl.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))
    tl.store(out_ptr + offsets, gelu_val, mask=mask)

@torch.fx.wrap
def triton_gelu(x):
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    out = torch.empty_like(x)
    gelu_kernel[grid](x, out, n_elements, BLOCK_SIZE)
    return out

@torch.fx.wrap
def optimized_gelu_mean(x):
    """Simple optimized version that uses Triton for GELU and regular mean"""
    gelu_out = triton_gelu(x)
    mean_out = gelu_out.mean((2, 3), keepdim=True)
    return (gelu_out, mean_out)

# Replacement function - must return a zero-argument function
def replacement_func():
    return optimized_gelu_mean