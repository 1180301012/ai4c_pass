import torch
import triton
import triton.language as tl

def pattern(x):
    # Match normalization operation directly
    # This should be the actual operation pattern in the graph
    return torch.nn.functional.normalize(x, p=2, dim=1)

def replacement_args(x):
    # For normalization replacement
    return (x,)

# High-performance L2 normalization kernel
@triton.jit
def l2_normalize_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Each program handles one column of the matrix
    col_id = tl.program_id(0)
    
    # Calculate row offsets for this column
    row_offsets = tl.arange(0, BLOCK_SIZE_M)
    mask = row_offsets < n_rows
    
    # Load the entire column
    x_col = tl.load(x_ptr + col_id * n_cols + row_offsets, mask=mask, other=0.0)
    
    # Compute L2 norm along the column
    sum_sq = tl.sum(x_col * x_col)
    norm = tl.sqrt(sum_sq)
    
    # Avoid division by zero
    norm = tl.where(norm > 1e-8, norm, 1.0)
    
    # Normalize the column
    out_col = x_col / norm
    
    # Store the normalized column
    tl.store(out_ptr + col_id * n_cols + row_offsets, out_col, mask=mask)

@torch.fx.wrap
def optimized_l2_normalize(x):
    # Get tensor dimensions
    n_rows = x.shape[0]
    n_cols = x.shape[1]
    
    # Set block sizes based on typical GPU configurations
    BLOCK_SIZE_M = 256  # Number of rows per block
    BLOCK_SIZE_N = 1    # Number of columns per program (one column per kernel)
    
    # Calculate number of programs needed (one per column)
    num_cols = n_cols
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Launch the kernel
    l2_normalize_kernel[(num_cols,)](
        x_ptr=x,
        out_ptr=out,
        n_rows=n_rows,
        n_cols=n_cols,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return out

def replacement_func():
    return optimized_l2_normalize