import torch
import triton
import triton.language as tl

# Pattern matching function
# This pattern matches:
#   tmp_0 = in_0.to(torch.float32)
#   tmp_1 = torch.tensor(1.0, dtype=torch.float32)
#   tmp_2 = tmp_1 - tmp_0  (call_function: torch.sub)
#   tmp_3 = tmp_2.to(torch.bool)
#   tmp_4 = tmp_2.masked_fill(tmp_3, -3.4028234663852886e+38)
#   return (tmp_4,)

def pattern(in_0):
    """
    Try matching just the masked_fill portion - simplified.
    """
    # Only match: to(bool) -> masked_fill
    tmp_2 = in_0
    tmp_3 = tmp_2.to(torch.bool)
    tmp_4 = tmp_2.masked_fill(tmp_3, -3.4028234663852886e+38)
    return tmp_4

# Argument extraction function
def replacement_args(in_0):
    # Extract and return arguments needed for the replacement
    return (in_0,)

# Optimized Triton kernel
@triton.jit
def fuse_masked_fill_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    NEG_INF: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for to(bool) -> masked_fill pattern.
    The input is already the result of (1.0 - original_input) in float32.
    We just need to:
    1. Convert to bool (non-zero -> True)
    2. Apply masked_fill with -inf for True positions
    """
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to ensure we don't go out of bounds
    
    # Load input (already float32, result of subtraction)
    input_val = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Create mask: to(torch.bool) converts non-zero to True
    mask_bool = input_val != 0.0
    
    # Apply masked_fill: replace True positions with -inf
    result = tl.where(mask_bool, NEG_INF, input_val)
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def fuse_masked_fill_wrapper(in_0):
    """
    Wrapper function that launches the Triton kernel.
    Replaces the to(bool) -> masked_fill pattern.
    """
    # Get number of elements
    n_elements = in_0.numel()
    
    # Choose block size - use 1024 for good occupancy
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # The value to fill with
    NEG_INF = -3.4028234663852886e+38
    
    # Allocate output tensor
    output = torch.empty_like(in_0, dtype=torch.float32)
    
    # Launch kernel
    fuse_masked_fill_kernel[(num_programs,)](
        input_ptr=in_0,
        output_ptr=output,
        n_elements=n_elements,
        NEG_INF=NEG_INF,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    """Return the replacement function"""
    return fuse_masked_fill_wrapper