import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    # Simple pattern: use both inputs
    return in_0 + in_1

def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Simple kernel that adds two tensors
@triton.jit
def simple_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)

@torch.fx.wrap
def simple_add(x, y):
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    out = torch.empty_like(x)
    simple_add_kernel[(num_programs,)](x, y, out, n_elements, BLOCK_SIZE)
    return out

def replacement_func():
    return simple_add