import torch
import triton
import triton.language as tl

def pattern(tmp_12):
    # Match the sparse matrix construction from model.py
    tmp_22 = torch.zeros(size=(1025, 1025), dtype=torch.int64)
    tmp_23 = tmp_12.sum(-1);  tmp_12 = None
    tmp_22[(slice(1, None, None), slice(1, None, None))] = tmp_23;  setitem_3 = tmp_22;  tmp_23 = setitem_3 = None
    tmp_22[(0, slice(0, None, None))] = 3969;  setitem_4 = tmp_22;  setitem_4 = None
    tmp_22[(slice(0, None, None), 0)] = 3970;  setitem_5 = tmp_22;  setitem_5 = None
    tmp_22[(0, 0)] = 3971;  setitem_6 = tmp_22;  setitem_6 = None
    tmp_28 = tmp_22.view(-1);  tmp_22 = None
    return tmp_28

@triton.jit
def optimized_sparse_matrix_kernel(
    output_ptr,
    input_ptr,
    matrix_size,
    boundary_size,
    val1,
    val2,
    val3,
    BLOCK_SIZE: tl.constexpr,
):
    matrix_size_sq = matrix_size * matrix_size
    
    # Compute the internal matrix size (excluding boundaries)
    internal_size = matrix_size - 1
    
    # Each program handles a block of data
    program_id = tl.program_id(0)
    total_elements = matrix_size_sq
    elements_per_block = BLOCK_SIZE
    
    start_idx = program_id * elements_per_block
    end_idx = min(start_idx + elements_per_block, total_elements)
    
    for idx in range(start_idx, end_idx):
        row = idx // matrix_size
        col = idx % matrix_size
        
        # Calculate the value to store
        if row > 0 and col > 0:
            # Internal elements - get from input tensor
            input_idx = (row - 1) * internal_size + (col - 1)
            value = tl.load(input_ptr + input_idx)
        elif row == 0 and col == 0:
            # Corner element
            value = val3
        elif row == 0:
            # Top boundary
            value = val1
        elif col == 0:
            # Left boundary  
            value = val2
        else:
            # Should not happen
            value = 0
        
        # Store the result
        tl.store(output_ptr + idx, value)

@torch.fx.wrap
def optimized_sparse_matrix_construction(tmp_12, matrix_size, boundary_size, val1, val2, val3):
    # Sum the input tensor along the last dimension
    tmp_23 = tmp_12.sum(-1)
    
    # Create output tensor
    matrix_size_sq = matrix_size * matrix_size
    tmp_28 = torch.empty(matrix_size_sq, dtype=torch.int64, device=tmp_12.device)
    
    # Launch optimized kernel
    BLOCK_SIZE = 1024
    num_programs = (matrix_size_sq + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    optimized_sparse_matrix_kernel[(num_programs,)](
        output_ptr=tmp_28,
        input_ptr=tmp_23,
        matrix_size=matrix_size,
        boundary_size=boundary_size,
        val1=val1,
        val2=val2,
        val3=val3,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return tmp_28

def replacement_args(tmp_12):
    matrix_size = 1025  # Size of the output matrix
    boundary_size = 1024  # Size of the internal matrix
    val1 = 3969  # Top boundary value
    val2 = 3970  # Left boundary value
    val3 = 3971  # Corner value
    return tmp_12, matrix_size, boundary_size, val1, val2, val3

def replacement_func():
    return optimized_sparse_matrix_construction