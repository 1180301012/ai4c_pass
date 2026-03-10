import torch
import triton
import triton.language as tl

# Pattern matching function - matches view(-1) operation for flattening
def pattern(x, y):
    # x should be the tensor being flattened
    # y is not used in the view but is needed for pattern matching
    result = x.view(-1)
    return result

# Argument extraction function
def replacement_args(x, y):
    return (x, y)

# Optimized view kernel for flattening large tensors
@triton.jit
def optimized_flatten_kernel(
    input_ptr,      # Input tensor pointer
    output_ptr,     # Output tensor pointer
    n_elements,     # Total number of elements to copy
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data in row-major order (natural for PyTorch tensors)
    input_val = tl.load(input_ptr + offsets, mask=mask, other=0)
    
    # Store to output (flattened layout)
    tl.store(output_ptr + offsets, input_val, mask=mask)

# Kernel wrapper optimized for flattening operations
@torch.fx.wrap
def triton_flatten(x):
    """
    Optimized flattening operation that ensures efficient memory access patterns
    for large tensors, particularly beneficial for int64 tensors that are commonly
    used in position index operations in transformer models.
    """
    N = x.numel()
    
    # Use different block sizes based on tensor size for optimal performance
    if N > 1000000:  # Large tensors
        BLOCK_SIZE = 2048
    elif N > 100000:  # Medium tensors
        BLOCK_SIZE = 1024
    else:  # Small tensors
        BLOCK_SIZE = 512
        
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor with same properties
    out = torch.empty(N, dtype=x.dtype, device=x.device)
    
    # Launch optimized flatten kernel
    optimized_flatten_kernel[(num_programs,)](
        input_ptr=x,
        output_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function returns the optimized flattening kernel
def replacement_func():
    return triton_flatten