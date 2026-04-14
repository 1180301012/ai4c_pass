import torch
import triton
import triton.language as tl

# Pattern matching function for scale * relu_result
def pattern(scale, relu_result):
    return scale * relu_result

# Argument extraction function  
def replacement_args(*args):
    # Pattern needs 2 arguments: (scale, relu_result)
    return (args[0], args[1])

@triton.jit
def multiplication_kernel(
    scale_ptr, relu_result_ptr, output_ptr,
    n_elements, BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    scale = tl.load(scale_ptr)  # scalar scale
    relu_vals = tl.load(relu_result_ptr + offsets, mask=mask, other=0.0)
    
    # Simple multiplication: scale * relu_result
    result = scale * relu_vals
    
    tl.store(output_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_multiplication(scale, relu_result):
    """Fused multiplication kernel: scale * relu_result"""
    n_elements = relu_result.numel()
    if n_elements == 0:
        return torch.zeros_like(relu_result)
    
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty_like(relu_result)
    
    multiplication_kernel[(num_programs,)](
        scale, relu_result, output,
        n_elements, BLOCK_SIZE
    )
    
    return output

# Replacement function
def replacement_func():
    return fused_multiplication