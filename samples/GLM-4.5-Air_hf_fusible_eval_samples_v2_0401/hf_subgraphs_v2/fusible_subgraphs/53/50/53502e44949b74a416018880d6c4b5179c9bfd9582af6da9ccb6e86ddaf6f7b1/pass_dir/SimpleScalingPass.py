import torch
import triton
import triton.language as tl

# Pattern matching function - matches a simple scaling operation
def pattern(tensor, scalar):
    return tensor * scalar

# Argument extraction function
def replacement_args(tensor, scalar):
    return (tensor, scalar)

# Triton kernel for scalar multiplication
@triton.jit
def scalar_mul_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    scalar_val,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Apply scaling
    out = x * scalar_val
    
    # Store result
    tl.store(output_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def scalar_multiply(input_tensor, scalar_val):
    output = torch.empty_like(input_tensor)
    
    n_elements = input_tensor.numel()
    BLOCK_SIZE = 256
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    scalar_mul_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output,
        n_elements=n_elements,
        scalar_val=scalar_val,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return scalar_multiply