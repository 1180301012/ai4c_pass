import torch
import triton
import triton.language as tl

def pattern(x):
    # Match the slice + fill_1 operations
    # x is the input tensor being sliced
    sliced = x[slice(None, None, None), slice(-5, None, None), slice(None, None, None)]
    result = sliced.fill_(1)
    return result

def replacement_args(x):
    return (x,)

@triton.jit
def slice_fill_kernel_1(
    tmp_ptr,
    out_ptr,
    n_rows, n_cols, n_slices,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    total_elements = n_rows * n_cols * n_slices
    
    mask = offsets < total_elements
    row_idx = (offsets // (n_cols * n_slices)) % n_rows
    col_idx = (offsets // n_slices) % n_cols
    slice_idx = offsets % n_slices
    
    # Check if we're in the last 5 rows
    in_target_region = (row_idx >= n_rows - 5)
    
    tl.store(out_ptr + offsets, in_target_region, mask=mask)

@triton.jit  
def slice_fill_kernel_2(
    tmp_ptr,
    out_ptr, 
    n_rows, n_cols, n_slices,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    total_elements = n_rows * n_cols * n_slices
    
    mask = offsets < total_elements
    row_idx = (offsets // (n_cols * n_slices)) % n_rows
    col_idx = (offsets // n_slices) % n_cols
    slice_idx = offsets % n_slices
    
    # Check if we're in the last 5 columns
    in_target_region = (col_idx >= n_cols - 5)
    
    tl.store(out_ptr + offsets, in_target_region, mask=mask)

@torch.fx.wrap
def optimized_slice_fill_1(tmp_0):
    n_rows, n_cols, n_slices = tmp_0.shape[1], tmp_0.shape[2], tmp_0.shape[0] * tmp_0.shape[3] if len(tmp_0.shape) > 3 else 1
    if len(tmp_0.shape) == 3:
        n_slices = 1
    
    total_elements = tmp_0.numel()
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.zeros_like(tmp_0, dtype=torch.bool)
    
    slice_fill_kernel_1[(num_programs,)](
        tmp_0,
        out,
        n_rows, n_cols, n_slices,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out.float().fill_(1)

@torch.fx.wrap
def optimized_slice_fill_2(tmp_0):
    n_rows, n_cols, n_slices = tmp_0.shape[1], tmp_0.shape[2], tmp_0.shape[0] * tmp_0.shape[3] if len(tmp_0.shape) > 3 else 1
    if len(tmp_0.shape) == 3:
        n_slices = 1
    
    total_elements = tmp_0.numel()
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.zeros_like(tmp_0, dtype=torch.bool)
    
    slice_fill_kernel_2[(num_programs,)](
        tmp_0,
        out,
        n_rows, n_cols, n_slices,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out.float().fill_(1)

def replacement_func():
    # Return the optimized kernel for slice on dimension 1 (rows)
    return optimized_slice_fill_1