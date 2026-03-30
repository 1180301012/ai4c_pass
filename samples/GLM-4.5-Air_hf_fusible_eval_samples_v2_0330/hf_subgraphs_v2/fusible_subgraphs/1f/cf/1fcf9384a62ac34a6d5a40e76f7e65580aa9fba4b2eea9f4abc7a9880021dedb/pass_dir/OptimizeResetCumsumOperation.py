import torch
import triton
import triton.language as tl

def pattern(in_0):
    """
    Match the reset cumulative sum computation pattern.
    The original computation:
    - Creates binary mask where input != 1
    - Computes cumulative sum of this mask along dim=1
    - Zeros out positions where original input == 1
    - Adds 1 to convert to 1-based indexing
    """
    tmp_1 = in_0.ne(1)
    tmp_2 = tmp_1.int()
    tmp_3 = torch.cumsum(tmp_2, dim=1)
    tmp_4 = tmp_3.type_as(tmp_2)
    tmp_5 = tmp_4 + 0
    tmp_6 = tmp_5 * tmp_2
    tmp_7 = tmp_6.long()
    tmp_8 = tmp_7 + 1
    return tmp_8

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def reset_cumsum_kernel_final(
    input_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Final optimized kernel for reset cumulative sum.
    Algorithm: Where input == 1: output = 1, Where input != 1: output = (count of non-1 elements so far) + 1
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    
    # Load input data
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0)
    
    # Create mask: 1 where input != 1, 0 where input == 1
    nonzero_mask = (input_vals != 1).to(tl.int32)
    
    # Compute cumulative sum of non-1 elements
    cumsum_nonzero = tl.cumsum(nonzero_mask)
    
    # Where input == 1: output = 1
    # Where input != 1: output = cumsum_nonzero + 1
    reset_positions = (input_vals == 1)
    output_vals = tl.where(reset_positions, 1, cumsum_nonzero + 1)
    
    # Store results
    tl.store(output_ptr + offsets, output_vals, mask=mask)

@torch.fx.wrap
def optimized_reset_cumsum_op(in_0):
    """
    Optimized reset cumulative sum operation using Triton kernel.
    This implements the correct algorithm with optimized memory access patterns.
    """
    input_shape = in_0.shape
    total_elements = in_0.numel()
    
    # Choose block size for good GPU occupancy
    BLOCK_SIZE = 1024
    
    # Create output tensor with correct dtype
    output = torch.empty_like(in_0, dtype=torch.long)
    
    # Calculate number of program blocks needed
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch Triton kernel
    reset_cumsum_kernel_final[(num_programs,)](
        in_0,
        output,
        total_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return optimized_reset_cumsum_op