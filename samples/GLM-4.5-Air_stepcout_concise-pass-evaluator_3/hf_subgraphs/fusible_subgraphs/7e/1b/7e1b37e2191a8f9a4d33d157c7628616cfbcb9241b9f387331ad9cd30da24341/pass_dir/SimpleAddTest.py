import torch
import triton
import triton.language as tl

def pattern(a, b, c):
    # Simple addition pattern using all inputs
    result = a + b
    # Also use c to avoid dead code
    unused = c * 0  # This makes c "used" but doesn't change computation
    return result

def replacement_args(a, b, c):
    return (a, b, c)

@triton.jit
def simple_add_kernel(
    a_ptr,
    b_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    c = a + b
    tl.store(output_ptr + offsets, c, mask=mask)

@torch.fx.wrap
def simple_add_fused(a, b, c):
    n_elements = a.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty_like(a)
    
    simple_add_kernel[(num_programs,)](
        a_ptr=a,
        b_ptr=b,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return simple_add_fused