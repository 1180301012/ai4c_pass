import torch
import triton
import triton.language as tl
from torch import device

def pattern(input_tensor, mask, fill_value):
    return input_tensor.masked_fill(mask, fill_value)

def replacement_args(input_tensor, mask, fill_value):
    return (input_tensor, mask, fill_value)

@triton.jit
def masked_fill_kernel(
    input_ptr,
    mask_ptr,
    output_ptr,
    fill_value,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    mask_vals = tl.load(mask_ptr + offsets, mask=mask, other=False)
    
    # Apply masked_fill: use fill_value where mask is True, otherwise use input value
    out = tl.where(mask_vals, fill_value, input_vals)
    
    tl.store(output_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_masked_fill(input_tensor, mask, fill_value):
    n_elements = input_tensor.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(input_tensor)
    
    masked_fill_kernel[(num_programs,)](
        input_ptr=input_tensor,
        mask_ptr=mask,
        output_ptr=out,
        fill_value=float(fill_value),
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

def replacement_func():
    return optimized_masked_fill