import torch
import triton
import triton.language as tl

@triton.jit
def simple_fma_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    c = tl.load(c_ptr + offsets, mask=mask, other=0.0)
    
    # Fused multiply-add: a * b + c
    result = a * b + c
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def simple_fma(a, b, c):
    n_elements = a.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty_like(a)
    
    simple_fma_kernel[(num_programs,)](
        a_ptr=a,
        b_ptr=b,
        c_ptr=c,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def pattern(a, b, c):
    """Simple element-wise multiply-add pattern"""
    result = a * b
    result += c
    return result

def replacement_args(a, b, c):
    return (a, b, c)

def replacement_func():
    return simple_fma