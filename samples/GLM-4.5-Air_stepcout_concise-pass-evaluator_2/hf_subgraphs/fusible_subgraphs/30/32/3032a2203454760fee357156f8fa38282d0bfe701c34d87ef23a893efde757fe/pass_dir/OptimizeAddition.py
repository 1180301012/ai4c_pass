import torch
import triton
import triton.language as tl

# Create a simple pass that matches the addition operation but uses all inputs to avoid dead code
def pattern(in_0, in_1, in_2, in_3):
    # Simple addition: this matches the core computation tmp_2 = in_2 + in_3
    # Note: in_0 and in_1 are not used in this specific pattern, but they are used
    # elsewhere in the computation (layer_norm), so they are alive in the graph
    return in_2 + in_3

def replacement_args(in_0, in_1, in_2, in_3):
    # For addition, we only need the two tensors being added
    return (in_2, in_3)

@triton.jit
def optimized_add_kernel(
    a_ptr,
    b_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    
    # Perform addition
    out = a + b
    
    # Store result
    tl.store(output_ptr + offsets, out, mask=mask)

@triton.jit
def optimized_add_kernel_autotuned(
    a_ptr,
    b_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Autotuned version with better performance for specific sizes"""
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    
    # Perform addition
    out = a + b
    
    # Store result
    tl.store(output_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_addition(tensor_a, tensor_b):
    # Ensure tensors are on the same device and have compatible shapes
    if tensor_a.device != tensor_b.device:
        tensor_b = tensor_b.to(tensor_a.device)
    
    # Calculate total number of elements
    n_elements = tensor_a.numel()
    
    # Choose optimal block size based on tensor size
    if n_elements > 1000000:
        BLOCK_SIZE = 2048
    elif n_elements > 100000:
        BLOCK_SIZE = 1024
    else:
        BLOCK_SIZE = 512
    
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output_tensor = torch.empty_like(tensor_a)
    
    # Launch kernel
    optimized_add_kernel_autotuned[(num_programs,)](
        tensor_a,
        tensor_b,
        output_tensor,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output_tensor

def replacement_func():
    return optimized_addition