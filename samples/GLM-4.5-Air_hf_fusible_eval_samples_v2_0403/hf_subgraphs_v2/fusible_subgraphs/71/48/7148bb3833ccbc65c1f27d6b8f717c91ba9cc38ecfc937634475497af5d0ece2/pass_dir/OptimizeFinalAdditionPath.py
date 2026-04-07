import torch
import triton
import triton.language as tl

def pattern(tmp_2, tmp_7):
    tmp_8 = tmp_2 + tmp_7
    tmp_2 = tmp_7 = None
    return tmp_8

def replacement_args(tmp_2, tmp_7):
    return (tmp_2, tmp_7)

@triton.jit
def simple_add_kernel(
    a_ptr,
    b_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    
    # Add
    out = a + b
    
    # Store
    tl.store(output_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def simple_add_torch(a, b):
    N = a.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty_like(a)

    simple_add_kernel[(num_programs,)](
        a_ptr=a,
        b_ptr=b,
        output_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out

def replacement_func():
    return simple_add_torch