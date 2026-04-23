import torch
import triton
import triton.language as tl

def pattern(x, device):
    return x.to(device=device, dtype=torch.bool)

def replacement_args(x, device):
    return (x, device)

@triton.jit
def to_bool_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    out = (x != 0)
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_to_bool(x, device):
    n_elements = x.numel()
    out = torch.empty_like(x, dtype=torch.bool)
    
    BLOCK_SIZE = 256
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    to_bool_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_to_bool