import torch
import triton
import triton.language as tl

# Pattern matching function - matches tensor addition operation
def pattern(tensor1, tensor2):
    return tensor1 + tensor2

# Argument extraction function  
def replacement_args(tensor1, tensor2):
    return (tensor1, tensor2)

# Triton kernel for optimized addition with autotune potential
@triton.jit
def optimized_add_kernel(
    input1_ptr,
    input2_ptr, 
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data from both tensors
    x = tl.load(input1_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(input2_ptr + offsets, mask=mask, other=0.0)
    
    # Perform addition
    out = x + y
    
    # Store result
    tl.store(output_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_add(input1, input2):
    # Handle the case where one argument might be a scalar
    if isinstance(input2, (int, float)):
        # Convert scalar to tensor for consistent processing
        scalar_val = input2
        input2 = torch.full_like(input1, scalar_val)
    elif isinstance(input1, (int, float)):
        scalar_val = input1
        input1 = torch.full_like(input2, scalar_val)
    
    # Ensure both tensors have the same shape
    if input1.shape != input2.shape:
        # This might happen in some cases, so broadcast if needed
        input1 = input1.expand_as(input2)
    
    output = torch.empty_like(input1)
    
    n_elements = input1.numel()
    BLOCK_SIZE = 1024  # Larger block size for better GPU utilization
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    optimized_add_kernel[(num_programs,)](
        input1_ptr=input1,
        input2_ptr=input2,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return optimized_add