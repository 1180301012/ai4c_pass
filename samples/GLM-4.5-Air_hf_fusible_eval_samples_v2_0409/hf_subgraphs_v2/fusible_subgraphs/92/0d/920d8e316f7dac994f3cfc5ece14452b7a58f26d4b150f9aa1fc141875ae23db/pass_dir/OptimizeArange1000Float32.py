import torch
from torch import device
import triton
import triton.language as tl

def pattern():
    tmp_0 = torch.arange(0, 1000, device=device(type='cuda'))
    tmp_1 = tmp_0.view(1, -1)
    tmp_0 = None
    tmp_2 = tmp_1.repeat(2, 1)
    tmp_1 = None
    return (tmp_2,)

def replacement_args():
    return ()

@triton.jit
def optimized_arange_1000_kernel(
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel for arange(0, 1000) repeated 2 times"""
    # Total elements: 1000 columns * 2 rows = 2000 elements
    total_elements = 2000
    
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask to ensure we don't go out of bounds
    mask = offsets < total_elements
    
    # Using modulo to create the repeated pattern [0, 1, 2, ..., 999] repeated 2 times
    col_index = offsets % 1000
    
    # Load the output pointer with calculated values
    tl.store(out_ptr + offsets, col_index, mask=mask)

@torch.fx.wrap
def optimized_arange_1000_repeat_float32():
    """Directly create a (2, 1000) tensor with float32 dtype"""
    n_rows = 2
    n_cols = 1000
    n_elements = n_rows * n_cols  # 2000
    
    # Create output tensor with float32 dtype
    out = torch.empty((n_rows, n_cols), device='cuda', dtype=torch.float32)
    
    # Triton kernel configuration - use power of 2 for efficiency
    BLOCK_SIZE = 1024  # Efficient GPU block size
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the optimized kernel
    optimized_arange_1000_kernel[(num_programs,)](
        out_ptr=out,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_arange_1000_repeat_float32