import torch
import triton
import triton.language as tl

# Pattern matching for simple division and type conversion
def pattern(in_5, in_4):
    tmp_4 = in_5 / in_4
    tmp_5 = tmp_4.to(torch.float32)
    tmp_4 = None
    return tmp_5

# Argument extraction for simple pattern
def replacement_args(in_5, in_4):
    return (in_5, in_4)

# Simple Triton kernel for division and type conversion
@triton.jit
def simple_division_kernel(
    input_ptr,
    divisor_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input and divisor
    inputs = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    divisor = tl.load(divisor_ptr)
    
    # Perform division (Triton operations are already float32)
    result = inputs / divisor
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def simple_division_function(input_tensor, divisor_tensor):
    n_elements = input_tensor.numel()
    
    # Adaptive block size based on tensor size for better performance
    if n_elements < 10000:
        BLOCK_SIZE = 256  # Small tensors: smaller block size
    elif n_elements < 100000:
        BLOCK_SIZE = 512  # Medium tensors: medium block size  
    else:
        BLOCK_SIZE = 1024  # Large tensors: larger block size
    
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor with float32 dtype
    output = torch.empty_like(input_tensor, dtype=torch.float32)
    
    # Launch kernel
    simple_division_kernel[(num_programs,)](
        input_ptr=input_tensor,
        divisor_ptr=divisor_tensor,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return simple_division_function