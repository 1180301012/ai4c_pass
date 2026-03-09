import torch
import triton
import triton.language as tl

@triton.jit
def transpose_kernel_2d(
    x_ptr,
    out_ptr,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Optimized kernel for transposing last two dimensions of a 4D tensor
    This kernel swaps the last two dimensions (dims -1, -2)
    """
    # Program IDs for 2D grid
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute memory offsets for transposed access pattern
    row_start = pid_m * BLOCK_SIZE_M
    col_start = pid_n * BLOCK_SIZE_N
    
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_M)
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE_N)
    
    # Create mask to handle boundary conditions
    row_mask = row_offsets < n_rows
    col_mask = col_offsets < n_cols
    
    # Load data with transposed indexing
    # For transpose: out[m, n] = x[n, m]
    x_ptrs = x_ptr + col_offsets[:, None] * n_rows + row_offsets[None, :]
    x = tl.load(x_ptrs, mask=row_mask[:, None] & col_mask[None, :], other=0.0)
    
    # Store with non-transposed indexing
    out_ptrs = out_ptr + row_offsets[:, None] * n_cols + col_offsets[None, :]
    tl.store(out_ptrs, x, mask=row_mask[:, None] & col_mask[None, :])

@torch.fx.wrap  
def optimized_transpose_last_two_dims(x):
    """
    Optimized function to transpose the last two dimensions with smart execution
    """
    # Get tensor shape and properties
    original_shape = x.shape
    n_rows = original_shape[-2]  # Second last dimension
    n_cols = original_shape[-1]  # Last dimension
    
    # Compute total size of the transposed matrix
    n_elements = n_rows * n_cols
    
    # For small matrices, use PyTorch's highly optimized transpose
    # Threshold: use kernel only if we have enough work to justify overhead
    OVERHEAD_THRESHOLD = 16384  # Elements (128x128 matrix)
    
    if n_elements < OVERHEAD_THRESHOLD:
        # For small matrices, use PyTorch's optimized implementation
        return x.transpose(-1, -2)
    
    # Adaptive block sizing based on matrix dimensions
    # For large matrices, use larger blocks for better GPU utilization
    if n_rows >= 128 and n_cols >= 128:
        BLOCK_SIZE_M = 32
        BLOCK_SIZE_N = 32
    elif n_rows >= 64 and n_cols >= 64:
        BLOCK_SIZE_M = 16
        BLOCK_SIZE_N = 16
    else:
        BLOCK_SIZE_M = 8  
        BLOCK_SIZE_N = 8
    
    # Compute grid dimensions
    grid_m = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (n_cols + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Only launch kernel if we have sufficient work to justify overhead
    if grid_m * grid_n < 2:  # Need at least 2 warps to justify kernel launch
        return x.transpose(-1, -2)
    
    # Create output tensor with same properties as input
    out = torch.empty_like(x)
    
    # Launch kernel - transposing just the last two dimensions
    # Optimized for 4D tensors (common case in transformers)
    if len(original_shape) == 4:
        # Reshape to 2D for transposing, then reshape back
        # Flatten all dimensions except the last two
        batch_size = original_shape[0] * original_shape[1] * original_shape[2]
        x_2d = x.reshape(batch_size, n_rows, n_cols)
        out_2d = out.reshape(batch_size, n_rows, n_cols)
        
        transpose_kernel_2d[(grid_m, grid_n)](
            x_ptr=x_2d,
            out_ptr=out_2d,
            n_rows=n_rows,
            n_cols=n_cols,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )
    else:
        # For other dimensions, handle appropriately 
        transpose_kernel_2d[(grid_m, grid_n)](
            x_ptr=x,
            out_ptr=out,
            n_rows=n_rows,
            n_cols=n_cols,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )
    
    return out

def pattern(x):
    """
    Pattern to match: transpose of last two dimensions
    """
    return x.transpose(-1, -2)

def replacement_args(x):
    """
    Extract arguments for the optimized kernel
    """
    return (x,)

def replacement_func():
    """
    Return the optimized transpose function
    """
    return optimized_transpose_last_two_dims