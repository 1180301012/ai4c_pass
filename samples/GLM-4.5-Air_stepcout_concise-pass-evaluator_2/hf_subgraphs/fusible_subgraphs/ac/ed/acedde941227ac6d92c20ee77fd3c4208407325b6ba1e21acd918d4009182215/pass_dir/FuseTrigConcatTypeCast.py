import torch
import triton
import triton.language as tl

def pattern(in_2):
    tmp_2 = torch.cat((in_2, in_2), dim=-1)
    return tmp_2.cos()

def replacement_args(in_2):
    return (in_2,)

@triton.jit
def fused_cos_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    inputs = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute cosine
    cos_vals = tl.cos(inputs)
    
    # Store results
    tl.store(output_ptr + offsets, cos_vals, mask=mask)

@torch.fx.wrap
def fused_cosine(input_tensor):
    n_elements = input_tensor.numel()
    output = torch.empty_like(input_tensor)
    
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_cos_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_cosine