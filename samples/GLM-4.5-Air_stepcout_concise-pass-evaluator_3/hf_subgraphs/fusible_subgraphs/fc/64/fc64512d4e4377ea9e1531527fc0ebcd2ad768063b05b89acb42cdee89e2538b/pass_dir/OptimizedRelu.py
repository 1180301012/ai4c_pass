import torch
import triton
import triton.language as tl

# Pattern matching function for ReLU operation
def pattern(in_0):
    """
    Matches ReLU operation for optimization
    """
    tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
    return tmp_0

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Highly optimized ReLU kernel
@triton.jit
def optimized_relu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    High-performance ReLU kernel with optimized memory access
    ReLU: x = max(0, x)
    """
    # Each program handles a contiguous block of data with excellent memory coalescing
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to ensure we don't go out of bounds
    
    # Load input data with optimized memory access
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Simple but highly optimized ReLU operation
    relu_out = tl.maximum(x, 0.0)
    
    # Store result with optimized memory access
    tl.store(output_ptr + offsets, relu_out, mask=mask)

# Optimized kernel wrapper (must be decorated with @torch.fx.wrap)
@torch.fx.wrap
def optimized_relu_triton(input_tensor):
    """
    Highly optimized wrapper function that launches the ReLU kernel
    Uses optimal configuration for GPU performance
    """
    n_elements = input_tensor.numel()
    output = torch.empty_like(input_tensor)
    
    # Choose optimal block size for maximum GPU occupancy
    if n_elements < 2048:
        BLOCK_SIZE = 128
    elif n_elements < 8192:
        BLOCK_SIZE = 256
    elif n_elements < 32768:
        BLOCK_SIZE = 512
    else:
        BLOCK_SIZE = 1024
    
    # Calculate optimal grid size
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel with optimal configuration
    optimized_relu_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Replacement function (must return a callable function reference)
def replacement_func():
    return optimized_relu_triton