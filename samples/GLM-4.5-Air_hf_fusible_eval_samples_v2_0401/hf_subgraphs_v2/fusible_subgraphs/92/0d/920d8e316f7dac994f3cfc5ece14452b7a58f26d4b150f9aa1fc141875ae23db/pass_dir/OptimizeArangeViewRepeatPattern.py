import torch
import triton
import triton.language as tl
from torch import device

# Pattern matching function - matches the exact computation from the models
def pattern():
    """
    Matches the arange -> view -> repeat pattern used in all three target graphs.
    Creates a (2, N) tensor where both rows contain the same range [0, 1, ..., N-1].
    This matches the exact computation structure without symbolic parameters.
    """
    tmp_0 = torch.arange(0, 1000, device=device(type='cuda'))
    tmp_1 = tmp_0.view(1, -1)
    tmp_0 = None
    tmp_2 = tmp_1.repeat(2, 1)
    tmp_1 = None
    return (tmp_2,)

# Argument extraction function
def replacement_args():
    """
    No arguments needed since the pattern matches a fixed computation.
    The replacement function will handle the specific size internally.
    """
    return ()

# Optimized Triton kernel for creating (2, N) tensor with identical rows
@triton.jit
def optimized_arange_repeat_kernel(
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    High-performance kernel that directly creates a (2, N) tensor where
    both rows contain the same range [0, 1, ..., N-1].
    This avoids intermediate arange, view, and repeat operations.
    """
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Calculate the values - this gives us the range [0, 1, 2, ...]
    values = offsets
    
    # Store the values twice (for both rows) at the correct positions
    # Row 0: offsets 0 to N-1
    tl.store(out_ptr + offsets, values, mask=mask)
    # Row 1: offsets N to 2*N-1  
    tl.store(out_ptr + offsets + n_elements, values, mask=mask)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap  
def optimized_arange_repeat():
    """
    Optimized function that creates a (2, 1000) tensor where
    both rows contain the same range [0, 1, ..., 999].
    Avoids intermediate tensor allocations and operations.
    """
    n_elements = 1000  # Fixed size for this pattern
    total_elements = 2 * n_elements
    
    # Output on CUDA device using torch's device specification
    out = torch.empty((2, n_elements), dtype=torch.int32, device='cuda')
    
    # Configure kernel launch parameters
    BLOCK_SIZE = 1024  # Optimized block size for GPU
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the optimized kernel
    optimized_arange_repeat_kernel[(num_programs,)](
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    """
    Returns the optimized kernel wrapper function.
    """
    return optimized_arange_repeat