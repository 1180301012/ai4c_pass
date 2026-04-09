import torch
from torch import device
import triton
import triton.language as tl

def pattern():
    tmp_0 = torch.arange(0, 128, device=device(type='cuda'))
    tmp_1 = tmp_0.view(1, -1)
    tmp_0 = None
    tmp_2 = tmp_1.repeat(2, 1)
    tmp_1 = None
    return (tmp_2,)

def replacement_args():
    return ()

@triton.jit
def optimized_arange_128_kernel(
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel for arange(0, 128) repeated 2 times"""
    # Total elements: 128 columns * 2 rows = 256 elements
    total_elements = 256
    
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask to ensure we don't go out of bounds
    mask = offsets < total_elements
    
    # Using modulo to create the repeated pattern [0, 1, 2, ..., 127] repeated 2 times
    # Since 128 = 2^7, modulo is very efficient
    col_index = offsets % 128
    
    # Load the output pointer with calculated values
    tl.store(out_ptr + offsets, col_index, mask=mask)

@torch.fx.wrap
def optimized_arange_128_repeat_bfloat16():
    """Directly create a (2, 128) tensor with bfloat16 dtype"""
    n_rows = 2
    n_cols = 128
    n_elements = n_rows * n_cols  # 256
    
    # Create output tensor with bfloat16 dtype
    out = torch.empty((n_rows, n_cols), device='cuda', dtype=torch.bfloat16)
    
    # Triton kernel configuration
    BLOCK_SIZE = 256  # Use block size that divides evenly into 256
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the optimized kernel
    optimized_arange_128_kernel[(num_programs,)](
        out_ptr=out,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_arange_128_repeat_bfloat16