import torch
import triton
import triton.language as tl


# Pattern matching function - matches just the != 0 and masked_fill part
def pattern(in_0):
    """ 
    Pattern to match: in_0 != 0 followed by masked_fill
    Returns only tmp_1 (the masked_fill result)
    """
    tmp_0 = in_0 != 0
    tmp_1 = in_0.masked_fill(tmp_0, -1000.0)
    return tmp_1


# Argument extraction function
def replacement_args(in_0):
    return (in_0,)


# Optimized Triton kernel for masked_fill operation
@triton.jit
def masked_fill_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    FILL_VALUE: tl.constexpr,
):
    """
    Optimized kernel for masked_fill operation.
    Simple and efficient implementation.
    """
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    
    # Compute: is_not_zero = (x != 0)
    is_not_zero = x != 0.0
    
    # Compute masked_fill: where is_not_zero is True, fill with FILL_VALUE
    result = tl.where(is_not_zero, FILL_VALUE, x)
    
    # Store output
    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def optimized_masked_fill(in_0):
    """
    Wrapper function that launches the optimized Triton kernel.
    """
    n_elements = in_0.numel()
    
    # Use optimized block size for this tensor size
    # 882,091 elements - using 2048 blocks gives good occupancy
    BLOCK_SIZE = 2048
    
    # Calculate grid size
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate output tensor (same dtype as input)
    out = torch.empty_like(in_0, dtype=in_0.dtype)
    
    # Launch kernel
    masked_fill_kernel[(num_programs,)](
        in_ptr=in_0,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        FILL_VALUE=-1000.0,
    )
    
    return out


# Replacement function - returns the function reference
def replacement_func():
    return optimized_masked_fill