import torch
import triton
import triton.language as tl

# Pattern matching function - matches the two sequential addition operations
def pattern(in_0, in_2, in_3):
    """Matches the two sequential in-place addition operations:
    in_3 += in_0
    in_4 = in_3; in_4 += in_2
    Returns tmp_0 which is the result of both additions
    """
    in_3 += in_0  # First addition
    in_4 = in_3   # Copy result
    in_4 += in_2  # Second addition
    tmp_0 = in_4  # Result after both additions
    return tmp_0

# Argument extraction function
def replacement_args(in_0, in_2, in_3):
    """Extracts arguments needed for the fused addition kernel"""
    return (in_0, in_2, in_3)

# Optimized kernel for fused two additions
@triton.jit
def fused_add_kernel(
    in_0_ptr,
    in_2_ptr, 
    in_3_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel that performs two sequential additions efficiently"""
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to ensure we don't go out of bounds
    
    # Load all input tensors
    in_0 = tl.load(in_0_ptr + offsets, mask=mask, other=0.0)
    in_2 = tl.load(in_2_ptr + offsets, mask=mask, other=0.0)
    in_3 = tl.load(in_3_ptr + offsets, mask=mask, other=0.0)
    
    # Perform fused operations: in_3 += in_0, then result += in_2
    result = in_3 + in_0 + in_2
    
    # Store the final result
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_add(in_0, in_2, in_3):
    """Wrapper function to launch the fused addition kernel"""
    # Determine the number of elements
    n_elements = in_0.numel()
    
    # Set block size and grid size
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(in_0)
    
    # Launch kernel
    fused_add_kernel[(num_programs,)](
        in_0_ptr=in_0,
        in_2_ptr=in_2,
        in_3_ptr=in_3,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function (returns function reference)
def replacement_func():
    return fused_add