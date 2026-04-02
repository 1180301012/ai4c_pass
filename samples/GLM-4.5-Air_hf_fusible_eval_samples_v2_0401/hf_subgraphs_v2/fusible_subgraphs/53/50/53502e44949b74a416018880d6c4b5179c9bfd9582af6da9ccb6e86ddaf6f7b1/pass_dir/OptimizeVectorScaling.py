import torch
import triton
import triton.language as tl

# Pattern matching function - matches tensor * scalar for typical vector sizes
def pattern(tensor, scalar):
    return tensor * scalar

# Argument extraction function
def replacement_args(tensor, scalar):
    return (tensor, scalar)

# Triton kernel optimized for typical embedding vector sizes (like 256)
@triton.jit
def vector_scale_kernel(
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
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Apply scaling
    out = x * scalar_val
    
    # Store result
    tl.store(output_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_vector_scale(input_tensor, scalar_val):
    output = torch.empty_like(input_tensor)
    
    n_elements = input_tensor.numel()
    
    # Dynamically choose block size based on tensor size for better GPU utilization
    if n_elements >= 16384:  # Larger tensors benefit from larger blocks
        BLOCK_SIZE = 1024
    elif n_elements >= 4096:  # Medium tensors
        BLOCK_SIZE = 512
    else:  # Small tensors - minimize overhead
        BLOCK_SIZE = 256
    
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    vector_scale_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output,
        n_elements=n_elements,
        scalar_val=scalar_val,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return optimized_vector_scale