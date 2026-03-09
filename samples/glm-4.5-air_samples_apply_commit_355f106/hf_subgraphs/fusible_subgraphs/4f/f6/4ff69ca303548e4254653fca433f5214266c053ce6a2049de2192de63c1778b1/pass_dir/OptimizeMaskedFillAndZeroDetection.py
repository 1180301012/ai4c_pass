import torch
import triton
import triton.language as tl

# Simple pattern for testing - only optimize masked_fill
def pattern(in_0):
    return in_0.masked_fill(in_0 != 0, -1000.0)

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Optimized kernel using Triton
@triton.jit
def simple_masked_fill_kernel(
    in_ptr,
    filled_ptr,
    n_elements,
    fill_value: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute program offsets
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    in_vals = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    
    # Apply masked fill: fill non-zero positions with fill_value
    filled_vals = tl.where(in_vals == 0.0, in_vals, fill_value)
    
    # Store results
    tl.store(filled_ptr + offsets, filled_vals, mask=mask)

@torch.fx.wrap
def simple_optimized_masked_fill(in_0):
    # Get tensor properties
    n_elements = in_0.numel()
    
    # Create output tensor
    filled_tensor = torch.empty_like(in_0)
    
    # Optimize for workload size - use larger block size for small tensors
    BLOCK_SIZE = 4096
    
    # Calculate grid size
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the optimized kernel
    simple_masked_fill_kernel[(num_programs,)](
        in_ptr=in_0,
        filled_ptr=filled_tensor,
        n_elements=n_elements,
        fill_value=-1000.0,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return filled_tensor

# Replacement function
def replacement_func():
    return simple_optimized_masked_fill