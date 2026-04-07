import torch
import triton
import triton.language as tl

def pattern(in_0, divisor):
    tmp_1 = in_0 // divisor
    tmp_2 = torch.sym_sum([1, tmp_1])
    return tmp_2

def replacement_args(in_0, divisor):
    return (in_0, divisor)

@triton.jit
def fused_division_sum_kernel(
    in_0_ptr,
    out_ptr,
    n_elements,
    divisor,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input tensor
    x = tl.load(in_0_ptr + offsets, mask=mask, other=0)
    
    # Fuse division and addition: compute 1 + (x // divisor)
    result = 1 + (x // divisor)
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_division_sum(in_0, divisor):
    n_elements = in_0.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(in_0, dtype=torch.int64)
    
    fused_division_sum_kernel[(num_programs,)](
        in_0_ptr=in_0,
        out_ptr=out,
        n_elements=n_elements,
        divisor=divisor,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_division_sum