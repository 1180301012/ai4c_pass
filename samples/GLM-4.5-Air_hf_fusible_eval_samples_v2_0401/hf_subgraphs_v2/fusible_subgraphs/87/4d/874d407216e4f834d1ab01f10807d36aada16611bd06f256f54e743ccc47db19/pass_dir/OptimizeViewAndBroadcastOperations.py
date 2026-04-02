import torch
import triton
import triton.language as tl

def pattern(in_1, in_2):
    """
    Simple pattern to test: view(-1, 1) * matrix
    """
    tmp_0 = in_1.view(-1, 1)
    tmp_1 = tmp_0 * in_2
    return tmp_1

def replacement_args(in_1, in_2):
    return (in_1, in_2)

@triton.jit
def optimized_kernel(
    in_1_ptr,
    in_2_ptr,
    out_1_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Program id for matrix operations
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute global indices for this thread block
    row_idx = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col_idx = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks for boundary conditions
    row_mask = row_idx < n_rows
    col_mask = col_idx < n_cols
    
    # Load in_1 values for this block of rows
    # These will be broadcasted across columns
    in_1_vals = tl.load(in_1_ptr + row_idx, mask=row_mask, other=0.0)
    
    # Load corresponding portion of in_2 matrix
    # Create indices for matrix access: row_indices * n_cols + col_indices
    row_indices = row_idx[:, None]  # Shape: (BLOCK_SIZE_M, 1)
    col_indices = col_idx[None, :]  # Shape: (1, BLOCK_SIZE_N)
    flat_indices = row_indices * n_cols + col_indices
    
    in_2_vals = tl.load(in_2_ptr + flat_indices, 
                       mask=row_mask[:, None] & col_mask[None, :], 
                       other=0.0)
    
    # Compute multiplication with broadcasting
    # This simulates: in_1.view(-1, 1) * in_2
    # Reshape in_1 to column vector and multiply
    in_1_col = tl.reshape(in_1_vals, (BLOCK_SIZE_M, 1))
    out_vals = in_1_col * in_2_vals
    
    # Store results
    tl.store(out_1_ptr + flat_indices, out_vals, 
             mask=row_mask[:, None] & col_mask[None, :])

@torch.fx.wrap
def kernel_wrapper(in_1, in_2):
    # Get tensor properties
    n_rows = in_1.shape[0]  # This is the original vector length
    n_cols = in_2.shape[1]  # This comes from the matrix dimension
    
    # Create output tensor
    out_1_shape = (n_rows, n_cols)
    out_1 = torch.empty(out_1_shape, dtype=in_1.dtype, device=in_1.device)
    
    # Choose optimal block sizes based on tensor dimensions
    if n_cols <= 32:
        # For smaller column dimensions, use smaller blocks
        BLOCK_SIZE_M = 32
        BLOCK_SIZE_N = min(16, n_cols)
    else:
        # For larger column dimensions, use larger blocks
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_N = 32
    
    # Ensure we don't create too many warps for small tensors
    if n_rows <= 256:
        BLOCK_SIZE_M = 16
    if n_cols <= 16:
        BLOCK_SIZE_N = 8
    
    # Launch kernel with optimized block sizes
    grid_m = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (n_cols + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    optimized_kernel[(grid_m, grid_n)](
        in_1_ptr=in_1,
        in_2_ptr=in_2,
        out_1_ptr=out_1,
        n_rows=n_rows,
        n_cols=n_cols,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return out_1

def replacement_func():
    return kernel_wrapper