import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """
    Pattern: view(-1, 1) operations + expand_as(result)
    This pattern optimizes the expensive view and expand operations
    """
    tmp_2 = in_0.view((-1, 1))
    tmp_3 = tmp_2.expand_as(in_1)
    return tmp_3

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def optimized_kernel(
    in_0_ptr,
    in_1_ptr,
    out_3_ptr,
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
    
    # Load in_0 values (these will be broadcasted)
    in_0_vals = tl.load(in_0_ptr + row_idx, mask=row_mask, other=0.0)
    
    # Create output by broadcasting in_0 to match in_1's shape
    # This eliminates the expensive expand_as operation
    out_3_vals = tl.broadcast_to(in_0_vals[:, None], (BLOCK_SIZE_M, BLOCK_SIZE_N))
    
    # Create indices for storing
    row_indices = row_idx[:, None]
    col_indices = col_idx[None, :]
    flat_indices = row_indices * n_cols + col_indices
    
    # Store results
    tl.store(out_3_ptr + flat_indices, out_3_vals, 
             mask=row_mask[:, None] & col_mask[None, :])

@torch.fx.wrap
def kernel_wrapper(in_0, in_1):
    # Get tensor properties
    n_rows = in_0.shape[0]  # Original vector length
    n_cols = in_1.shape[1]  # From the matrix dimension
    
    # Create output tensor
    out_3_shape = (n_rows, n_cols)
    out_3 = torch.empty(out_3_shape, dtype=in_0.dtype, device=in_0.device)
    
    # Choose optimal block sizes
    if n_cols <= 32:
        BLOCK_SIZE_M = 32
        BLOCK_SIZE_N = min(16, n_cols)
    else:
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_N = 32
    
    # Adjust for small tensors
    if n_rows <= 256:
        BLOCK_SIZE_M = 16
    if n_cols <= 16:
        BLOCK_SIZE_N = 8
    
    # Launch kernel
    grid_m = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (n_cols + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    optimized_kernel[(grid_m, grid_n)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_3_ptr=out_3,
        n_rows=n_rows,
        n_cols=n_cols,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return out_3

def replacement_func():
    return kernel_wrapper