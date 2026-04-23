import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    return in_1 * torch.nn.functional.relu(in_2, inplace=False) + in_0

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def fused_element_wise_kernel(
    in_ptr,
    out_ptr,
    scale,
    bias,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    
    x_scaled = x * scale
    x_relu = tl.where(x_scaled > 0, x_scaled, 0.0)
    out = x_relu + bias
    
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_element_wise_wrapper(in_0, in_1, in_2):
    bias = in_0.item()
    scale = in_1.item()
    n_elements = in_2.numel()
    
    out = torch.empty_like(in_2)
    
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_element_wise_kernel[(num_programs,)](
        in_ptr=in_2,
        out_ptr=out,
        scale=scale,
        bias=bias,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_element_wise_wrapper