import torch
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
def optimized_arange_repeat_kernel(
    out_ptr,
    start_val,
    end_val,
    n_elements,
    rows_to_repeat,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel that creates repeated arange pattern directly"""
    # Calculate the total number of elements in the output tensor
    total_elements = (end_val - start_val) * rows_to_repeat
    
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask to ensure we don't go out of bounds
    mask = offsets < total_elements
    
    # For each position, calculate the value using division and modulo
    # Since each row contains the same values [start_val, start_val+1, ..., end_val-1]
    col_index = offsets % (end_val - start_val) + start_val
    
    # Load the output pointer with calculated values
    tl.store(out_ptr + offsets, col_index, mask=mask)

@torch.fx.wrap
def optimized_arange_repeat(start, end, dtype, rows=2):
    """Directly create a 2D tensor with repeated arange values"""
    n_cols = end - start
    n_rows = rows
    n_elements = n_rows * n_cols
    
    # Determine the Triton data type
    if dtype == torch.float32:
        tl_dtype = tl.float32
    elif dtype == torch.float16:
        tl_dtype = tl.float16
    elif dtype == torch.bfloat16:
        tl_dtype = tl.bfloat16
    else:
        # Fallback to torch dtype for unknown types
        tl_dtype = dtype
    
    # Create output tensor
    out = torch.empty((n_rows, n_cols), device='cuda', dtype=dtype)
    
    # Triton kernel configuration
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the optimized kernel
    optimized_arange_repeat_kernel[(num_programs,)](
        out_ptr=out,
        start_val=start,
        end_val=end,
        n_elements=n_elements,
        rows_to_repeat=n_rows,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_arange_repeat