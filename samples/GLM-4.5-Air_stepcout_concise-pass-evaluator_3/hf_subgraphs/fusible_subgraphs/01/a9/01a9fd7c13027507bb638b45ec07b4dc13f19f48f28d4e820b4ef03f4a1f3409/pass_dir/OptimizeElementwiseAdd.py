import torch
import triton
import triton.language as tl

# Pattern matching for tensor element-wise addition
def pattern(x, y):
    """
    Pattern: Element-wise addition between two tensors (not scalars/literals)
    This specifically matches tensor operations, not literal additions like 0 + tensor
    """
    # Element-wise addition between two tensors
    # Note: The pattern should be specific enough to avoid matching 0 + tmp_6
    result = x + y
    return result

# Argument extraction function  
def replacement_args(x, y):
    """
    Extract the two tensor arguments for element-wise addition
    """
    return x, y

# Optimized element-wise addition kernel
@triton.jit
def optimized_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to ensure we don't go out of bounds
    
    # Load both tensors
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Perform addition
    out = x + y
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@triton.jit
def optimized_add_kernel_autotuned(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to ensure we don't go out of bounds
    
    # Load both tensors
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Perform addition
    out = x + y
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_elementwise_add(x, y):
    """
    Optimized element-wise addition using Triton with adaptive block sizing
    
    Note: This function assumes both x and y are tensors, not scalars.
    The pattern matching should ensure this.
    """
    # Verify inputs are tensors (this should be guaranteed by pattern matching)
    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        # Fallback to regular addition if inputs are not tensors
        return x + y
    
    # Get total number of elements
    n_elements = x.numel()
    
    # Choose optimal block size based on tensor size
    if n_elements < 50000:
        # Very small tensors: use smaller block size
        BLOCK_SIZE = 256
    elif n_elements < 200000:
        # Small tensors: use medium block size to balance overhead and occupancy  
        BLOCK_SIZE = 512
    elif n_elements < 1000000:
        # Medium tensors: use larger block size for better GPU occupancy
        BLOCK_SIZE = 1024  
    else:
        # Large tensors: use even larger block size
        BLOCK_SIZE = 2048
    
    # Calculate number of programs needed
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Launch kernel
    optimized_add_kernel_autotuned[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return optimized_elementwise_add