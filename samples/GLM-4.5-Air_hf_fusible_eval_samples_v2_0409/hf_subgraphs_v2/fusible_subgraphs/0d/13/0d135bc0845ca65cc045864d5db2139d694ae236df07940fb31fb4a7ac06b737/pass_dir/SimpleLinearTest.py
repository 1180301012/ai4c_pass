import torch
import triton
import triton.language as tl

@triton.jit
def simple_linear_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Simple scaling (temporary while we debug)
    result = input_vals * 2.0
    
    # Add bias
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    result = result + bias
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def simple_linear(input, weight, bias):
    n_elements = input.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(input)
    
    simple_linear_kernel[(num_programs,)](
        input_ptr=input,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def pattern(input, weight, bias):
    # Simple linear operation for testing
    result = torch.nn.functional.linear(input, weight, bias)
    return result

def replacement_args(input, weight, bias):
    return (input, weight, bias)

def replacement_func():
    return simple_linear