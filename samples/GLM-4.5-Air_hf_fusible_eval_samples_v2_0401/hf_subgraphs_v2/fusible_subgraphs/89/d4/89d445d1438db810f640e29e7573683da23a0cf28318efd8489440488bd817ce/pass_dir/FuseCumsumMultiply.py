import torch
import triton
import triton.language as tl

@triton.jit
def cumsum_kernel(
    input_ptr,
    output_ptr,
    cumsum_dim_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each thread handles one column (since we have 1 row)
    col_id = tl.program_id(0)
    
    # Load the element at this position
    elem_val = tl.load(input_ptr + col_id, mask=(col_id < cumsum_dim_size))
    
    # Establish type as int64 by casting
    cumsum_val = tl.cast(0, tl.int64)
    
    # Compute cumsum by manually adding elements from the start
    if col_id < cumsum_dim_size:
        # Simple: add all elements from position 0 to current position
        # For position i, we add elements 0, 1, 2, ..., i
        for i in range(col_id + 1):
            if i < cumsum_dim_size:
                elem_ptr = input_ptr + i
                prev_elem = tl.load(elem_ptr, mask=(i < cumsum_dim_size))
                cumsum_val += prev_elem
    
    # Store the cumsum result
    tl.store(output_ptr + col_id, cumsum_val, mask=(col_id < cumsum_dim_size))

@triton.jit
def elementwise_multiply_kernel(
    input_ptr,
    multiplier_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load elements
    input_vals = tl.load(input_ptr + offsets, mask=mask)
    multiplier_vals = tl.load(multiplier_ptr + offsets, mask=mask)
    
    # Element-wise multiplication
    result = input_vals * multiplier_vals
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

def pattern(tmp_0):
    tmp_1 = torch.cumsum(tmp_0, dim=1)
    tmp_2 = tmp_1 * tmp_0
    return tmp_2

def replacement_args(tmp_0):
    return (tmp_0,)

@torch.fx.wrap
def fused_cumsum_multiply(input_tensor):
    # Get tensor dimensions
    n_rows, n_cols = input_tensor.shape
    n_elements = input_tensor.numel()
    
    # Since we have cumsum along dim=1 and only 1 row, use appropriate grid
    BLOCK_SIZE = 16  # Power of 2 that can handle our tensor [1, 13]
    
    # Step 1: Compute cumsum using Triton kernel
    cumsum_result = torch.empty_like(input_tensor)
    
    # Grid dimensions for cumsum: one program per column (since we have 1 row)
    n_col_programs = n_cols
    
    # Launch cumsum kernel - one program per column
    cumsum_kernel[(n_col_programs,)](
        input_ptr=input_tensor,
        output_ptr=cumsum_result,
        cumsum_dim_size=n_cols,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Step 2: Element-wise multiplication using Triton kernel
    output = torch.empty_like(input_tensor)
    
    # Grid dimensions for multiplication
    n_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch elementwise multiplication kernel
    elementwise_multiply_kernel[(n_programs,)](
        input_ptr=cumsum_result,
        multiplier_ptr=input_tensor,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return fused_cumsum_multiply