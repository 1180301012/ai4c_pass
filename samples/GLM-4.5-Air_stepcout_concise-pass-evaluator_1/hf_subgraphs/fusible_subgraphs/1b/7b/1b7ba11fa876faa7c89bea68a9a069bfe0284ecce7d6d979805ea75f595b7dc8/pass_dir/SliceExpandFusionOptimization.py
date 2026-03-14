import torch
import triton
import triton.language as tl

def pattern(tmp_1, slice_dim, slice_size, expand_dim0, expand_dim1):
    """
    Pattern matching for slice + expand operations
    tmp_2 = tmp_1[:, :slice_size]
    tmp_3 = tmp_2.expand(expand_dim0, expand_dim1)
    """
    tmp_2 = tmp_1[slice(None, None, None), slice(None, slice_size, None)]
    tmp_3 = tmp_2.expand(expand_dim0, expand_dim1)
    return tmp_3

def replacement_args(tmp_1, slice_size, expand_dim0, expand_dim1):
    return (tmp_1, slice_size, expand_dim0, expand_dim1)

@triton.jit
def slice_expand_kernel(
    input_ptr,
    output_ptr,
    input_dim0,
    input_dim1,
    slice_size,
    expand_dim0,
    expand_dim1,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel that directly creates expanded tensor without intermediate slice"""
    pid = tl.program_id(0)
    
    # Each program handles one row of the output tensor
    row_idx = pid
    
    if row_idx >= expand_dim0:
        return
    
    # Calculate column indices to process
    col_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < expand_dim1
    
    # Determine what to store in each position
    if row_idx < input_dim0 and col_offsets < min(slice_size, input_dim1):
        # Copy from input tensor where valid
        input_row_idx = row_idx
        input_col = col_offsets
        value = tl.load(input_ptr + input_row_idx * input_dim1 + input_col, mask=mask)
    else:
        # Fill with zeros (or the original expand behavior which broadcasts)
        value = tl.load(input_ptr + row_idx % input_dim0 * input_dim1 + col_offsets, mask=mask)
    
    # Directly write to output
    output_idx = row_idx * expand_dim1 + col_offsets
    tl.store(output_ptr + output_idx, value, mask=mask)

@torch.fx.wrap
def optimized_slice_expand(tmp_1, slice_size, expand_dim0, expand_dim1):
    """Optimized function that fuses slice and expand operations"""
    input_shape = tmp_1.shape
    input_dim0 = input_shape[0]
    input_dim1 = input_shape[1]
    
    # Allocate output tensor
    output_shape = (expand_dim0, expand_dim1)
    output = torch.empty(output_shape, dtype=tmp_1.dtype, device=tmp_1.device)
    
    # Calculate launch configuration
    n_elements = expand_dim0 * expand_dim1
    BLOCK_SIZE = 1024  # Optimal block size for most GPUs
    n_programs = (expand_dim0 + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Handle cases where we might need more programs
    if expand_dim1 > BLOCK_SIZE:
        n_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    slice_expand_kernel[(n_programs,)](
        input_ptr=tmp_1,
        output_ptr=output,
        input_dim0=input_dim0,
        input_dim1=input_dim1,
        slice_size=slice_size,
        expand_dim0=expand_dim0,
        expand_dim1=expand_dim1,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return optimized_slice_expand