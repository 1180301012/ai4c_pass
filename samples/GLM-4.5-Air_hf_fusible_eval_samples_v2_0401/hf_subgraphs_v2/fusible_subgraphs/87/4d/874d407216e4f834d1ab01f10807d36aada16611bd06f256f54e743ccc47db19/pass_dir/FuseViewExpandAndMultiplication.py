import torch
import triton
import triton.language as tl

def pattern(in_1, in_2, in_0):
    """
    Pattern: Optimized fusion of view(-1, 1), multiplication, and view(-1, 1) + expand_as
    This pattern fuses multiple operations to reduce memory overhead and improve cache efficiency
    """
    tmp_0 = in_1.view(-1, 1)
    tmp_1 = tmp_0 * in_2
    tmp_2 = in_0.view((-1, 1))
    tmp_3 = tmp_2.expand_as(tmp_1)
    return tmp_1, tmp_3

def replacement_args(in_1, in_2, in_0):
    return (in_1, in_2, in_0)

@triton.jit
def optimized_kernel(
    in_1_ptr,
    in_2_ptr,
    in_0_ptr,
    out_1_ptr,
    out_3_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Program id for matrix operations
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute global indices
    row_idx = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col_idx = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Masks for boundary conditions
    row_mask = row_idx < n_rows
    col_mask = col_idx < n_cols
    
    # Load in_1 and in_0 values for this block
    in_1_vals = tl.load(in_1_ptr + row_idx, mask=row_mask, other=0.0)
    in_0_vals = tl.load(in_0_ptr + row_idx, mask=row_mask, other=0.0)
    
    # Load in_2 matrix
    row_indices = row_idx[:, None]
    col_indices = col_idx[None, :]
    flat_indices = row_indices * n_cols + col_indices
    
    in_2_vals = tl.load(in_2_ptr + flat_indices, 
                       mask=row_mask[:, None] & col_mask[None, :], 
                       other=0.0)
    
    # fused computation:
    # tmp_1 = in_1.view(-1, 1) * in_2
    # tmp_3 = in_0.view(-1, 1).expand_as(tmp_1)
    in_1_col = tl.reshape(in_1_vals, (BLOCK_SIZE_M, 1))
    out_1_vals = in_1_col * in_2_vals
    out_3_vals = tl.broadcast_to(in_0_vals[:, None], (BLOCK_SIZE_M, BLOCK_SIZE_N))
    
    # Store both results
    tl.store(out_1_ptr + flat_indices, out_1_vals, 
             mask=row_mask[:, None] & col_mask[None, :])
    tl.store(out_3_ptr + flat_indices, out_3_vals, 
             mask=row_mask[:, None] & col_mask[None, :])

@torch.fx.wrap
def kernel_wrapper(in_1, in_2, in_0):
    # Get tensor properties
    n_rows = in_1.shape[0]
    n_cols = in_2.shape[1]
    
    # Create output tensors
    out_1_shape = (n_rows, n_cols)
    out_3_shape = (n_rows, n_cols)
    
    out_1 = torch.empty(out_1_shape, dtype=in_1.dtype, device=in_1.device)
    out_3 = torch.empty(out_3_shape, dtype=in_0.dtype, device=in_0.device)
    
    # Use larger block sizes for better efficiency with fused operations
    if n_rows <= 512 and n_cols <= 32:
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_N = 32
    else:
        BLOCK_SIZE_M = 128
        BLOCK_SIZE_N = 64
    
    # Adjust block sizes for small tensors
    if n_rows <= 256:
        BLOCK_SIZE_M = 32
    if n_cols <= 16:
        BLOCK_SIZE_N = 16
    
    # Launch kernel
    grid_m = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (n_cols + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    optimized_kernel[(grid_m, grid_n)](
        in_1_ptr=in_1,
        in_2_ptr=in_2,
        in_0_ptr=in_0,
        out_1_ptr=out_1,
        out_3_ptr=out_3,
        n_rows=n_rows,
        n_cols=n_cols,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return out_1, out_3

def replacement_func():
    return kernel_wrapper