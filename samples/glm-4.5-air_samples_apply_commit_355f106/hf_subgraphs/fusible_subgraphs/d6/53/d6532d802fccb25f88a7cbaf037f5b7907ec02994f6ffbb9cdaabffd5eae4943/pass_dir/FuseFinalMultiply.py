import torch
import triton
import triton.language as tl

def pattern(weight, normalized_input):
    tmp_17 = weight * normalized_input
    return tmp_17

def replacement_args(weight, normalized_input):
    return (weight, normalized_input)

@triton.jit
def fused_multiply_kernel(weight_ptr, normalized_ptr, output_ptr, 
                         n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load tensors
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    normalized = tl.load(normalized_ptr + offsets, mask=mask, other=0.0)
    
    # Multiply and store
    result = weight * normalized
    tl.store(output_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_multiply(weight, normalized_input):
    # Handle the case where weight might be broadcasted
    if weight.dim() != normalized_input.dim():
        # Expand weight to match dimensions
        broadcast_shape = list(normalized_input.shape)
        weight = weight.expand(broadcast_shape)
    
    n_elements = weight.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty_like(weight)
    
    fused_multiply_kernel[(num_programs,)](
        weight_ptr=weight,
        normalized_ptr=normalized_input,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return fused_multiply