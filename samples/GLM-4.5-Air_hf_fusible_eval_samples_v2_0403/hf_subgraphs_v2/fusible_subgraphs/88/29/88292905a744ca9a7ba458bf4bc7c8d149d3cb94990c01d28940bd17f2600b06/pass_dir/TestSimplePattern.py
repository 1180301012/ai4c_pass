import torch
import triton
import triton.language as tl

def pattern(a, b):
    return a + b

def replacement_args(a, b):
    return (a, b)

@triton.jit
def simple_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_elements: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= n_elements:
        return
    
    a_val = tl.load(a_ptr + pid)
    b_val = tl.load(b_ptr + pid)
    # Return a + b
    tl.store(out_ptr + pid, a_val + b_val)

@torch.fx.wrap
def simple_identity(a, b):
    n_elements = a.numel()
    out = torch.empty_like(a)
    simple_kernel[(n_elements,)](a, b, out, n_elements)
    return out

def replacement_func():
    return simple_identity