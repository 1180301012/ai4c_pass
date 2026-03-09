import torch
import triton
import triton.language as tl

def pattern(x):
    t1 = x - 0.25
    t2 = t1 * 3.141592653589793
    return t2

def replacement_args(x):
    return (x,)

@triton.jit
def elementwise_fused_kernel(
    input_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID and offset
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Apply fused operations: (x - 0.25) * π
    result = (x - 0.25) * 3.141592653589793
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def optimized_elementwise_ops(input_tensor):
    n_elements = input_tensor.numel()
    
    # Calculate grid size
    BLOCK_SIZE = 1024
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output = torch.empty_like(input_tensor)
    
    # Flatten input for efficient access
    input_flat = input_tensor.reshape(-1)
    output_flat = output.reshape(-1)
    
    # Launch kernel
    elementwise_fused_kernel[grid_size](
        input_ptr=input_flat,
        output_ptr=output_flat,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return optimized_elementwise_ops