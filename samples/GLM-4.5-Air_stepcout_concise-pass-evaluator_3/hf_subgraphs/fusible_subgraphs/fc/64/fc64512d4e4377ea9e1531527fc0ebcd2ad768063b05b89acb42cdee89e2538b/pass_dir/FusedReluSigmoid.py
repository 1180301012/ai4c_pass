import torch
import triton
import triton.language as tl

# Pattern matching function - optimized sigmoid operation
def pattern(in_0):
    """
    Matches Sigmoid operation for optimization
    """
    tmp_1 = torch.sigmoid(in_0)
    return tmp_1

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Specialized kernel optimized specifically for small tensors (low overhead)
@triton.jit
def small_tensor_sigmoid_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Specialized sigmoid kernel optimized for small tensors with minimal kernel launch overhead
    Uses the most computation-efficient approach for small workloads
    """
    # Minimal block size for small tensors to reduce overhead
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load all needed data at once
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Simple but highly efficient sigmoid computation for small tensors
    # Avoid complex branching for maximum speed in small workloads
    sigmoid_out = 1.0 / (1.0 + tl.exp(-x))
    
    # Store result
    tl.store(output_ptr + offsets, sigmoid_out, mask=mask)

# Specialized wrapper for small tensors with minimal overhead
@torch.fx.wrap
def specialized_small_tensor_sigmoid(input_tensor):
    """
    Specialized wrapper function optimized for small tensors with minimal kernel launch overhead
    Uses the most efficient configuration for the specific tensor sizes in this problem
    """
    n_elements = input_tensor.numel()
    output = torch.empty_like(input_tensor)
    
    # Optimal configuration specifically for our tensor sizes (256 and 8192 elements)
    if n_elements <= 512:
        BLOCK_SIZE = 256  # Fewer launches for very small tensors
    elif n_elements <= 4096:
        BLOCK_SIZE = 512
    else:  # n_elements <= 8192
        BLOCK_SIZE = 1024
    
    # Calculate minimal grid size
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the highly optimized small tensor kernel
    small_tensor_sigmoid_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Replacement function (must return a callable function reference)
def replacement_func():
    return specialized_small_tensor_sigmoid