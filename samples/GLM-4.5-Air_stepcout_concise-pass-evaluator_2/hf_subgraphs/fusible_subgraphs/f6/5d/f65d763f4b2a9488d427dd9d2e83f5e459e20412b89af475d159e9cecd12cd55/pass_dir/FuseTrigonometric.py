import torch
import triton
import triton.language as tl

@triton.jit
def cosine_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Triton kernel that computes cosine"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    cos_vals = tl.cos(x)
    tl.store(out_ptr + offsets, cos_vals, mask=mask)

@torch.fx.wrap
def triton_cosine(x):
    """Triton function that computes cosine"""
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    cosine_kernel[(num_programs,)](x, out, n_elements, BLOCK_SIZE)
    return out

def pattern(x):
    """Pattern that matches the cosine operation"""
    return x.cos()

def replacement_args(x):
    return (x,)

def replacement_func():
    """Replacement function that returns the Triton cosine function"""
    return triton_cosine