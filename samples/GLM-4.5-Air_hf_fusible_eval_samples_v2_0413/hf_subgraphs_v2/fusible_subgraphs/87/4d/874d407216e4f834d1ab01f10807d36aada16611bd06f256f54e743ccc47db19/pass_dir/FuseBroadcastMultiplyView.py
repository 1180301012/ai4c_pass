import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_1, in_2):
    tmp_0 = in_1.view(-1, 1)
    tmp_1 = tmp_0 * in_2
    return tmp_1

# Argument extraction function
def replacement_args(in_1, in_2):
    # Calculate the output shape: [in_1.shape[0], in_2.shape[1]]
    output_shape = (in_1.shape[0], in_2.shape[1])
    return (in_1, in_2, output_shape)

# Triton kernel for fused broadcast multiplication - with autotune configs
@triton.jit
def fused_broadcast_multiply_kernel(
    vector_ptr,
    matrix_ptr,
    output_ptr,
    vector_size,
    matrix_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a row of the output matrix (better memory coalescing)
    row_id = tl.program_id(0)
    block_start = tl.program_id(1) * BLOCK_SIZE
    col_offset = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for column bounds
    col_mask = col_offset < matrix_cols
    
    # Load the vector element for this row (same for all columns)
    vector_val = tl.load(vector_ptr + row_id, mask=row_id < vector_size)
    
    # Load the corresponding elements from the matrix row
    matrix_row = tl.load(matrix_ptr + row_id * matrix_cols + col_offset, mask=col_mask, other=0.0)
    
    # Perform multiplication (vector_val broadcasts to all columns)
    result = vector_val * matrix_row
    
    # Store the result to the output row
    tl.store(output_ptr + row_id * matrix_cols + col_offset, result, mask=col_mask)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def fused_broadcast_multiply(vector, matrix, output_shape):
    vector_size = vector.shape[0]
    matrix_cols = matrix.shape[1]
    
    # Use optimized block size for small problems
    BLOCK_SIZE = 128
    num_rows = vector_size
    num_blocks_cols = (matrix_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty(output_shape, dtype=vector.dtype, device=vector.device)
    
    fused_broadcast_multiply_kernel[(num_rows, num_blocks_cols)](
        vector_ptr=vector,
        matrix_ptr=matrix,
        output_ptr=output,
        vector_size=vector_size,
        matrix_cols=matrix_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_broadcast_multiply