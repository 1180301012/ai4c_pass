import torch
import triton
import triton.language as tl

def pattern(in_5):
    tmp_4 = in_5.to(torch.float32)
    return tmp_4

@triton.jit
def simple_kernel(
    in_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    in_vals = tl.load(in_ptr + offsets, mask=mask, other=0)
    
    # Convert to float32
    result = in_vals.to(tl.float32)
    
    # Store output
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def simple_conversion(in_5):
    n_elements = in_5.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(in_5, dtype=torch.float32)
    
    simple_kernel[(num_programs,)](
        in_ptr=in_5,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_args(in_5):
    return (in_5,)

def replacement_func():
    return simple_conversion