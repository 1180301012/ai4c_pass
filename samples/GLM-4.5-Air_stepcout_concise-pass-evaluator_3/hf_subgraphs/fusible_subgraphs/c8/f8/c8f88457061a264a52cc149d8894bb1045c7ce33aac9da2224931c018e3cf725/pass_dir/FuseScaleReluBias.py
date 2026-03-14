import torch
import triton
import triton.language as tl

def pattern(scale, relu_input):
    """Pattern: scale * relu_result"""
    result = scale * relu_input
    return result

def replacement_args(scale, relu_input):
    """Extract arguments for the replacement kernel"""
    return (scale, relu_input)

@triton.jit
def multiply_kernel(
    scale_ptr,
    relu_input_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple Triton kernel for scale * relu_input"""
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs with broadcast support for scalar scale
    scale = tl.load(scale_ptr + offsets, mask=mask, other=1.0)
    relu_input = tl.load(relu_input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute: scale * relu_input
    out = scale * relu_input
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def multiply_impl(scale, relu_input):
    """Implementation of scale * relu_input"""
    n_elements = relu_input.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(relu_input)
    
    multiply_kernel[(num_programs,)](
        scale_ptr=scale,
        relu_input_ptr=relu_input,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    """Returns the multiplication implementation function"""
    return multiply_impl