import torch
import triton
import triton.language as tl

def pattern(arg):
    """
    Simple pattern to match L2 normalization optimization
    tmp_0 = arg.norm(p = 2, dim = -1, keepdim = True)
    tmp_1 = arg / tmp_0
    Returns tmp_1 (normalized arg)
    """
    tmp_0 = arg.norm(p = 2, dim = -1, keepdim = True)
    tmp_1 = arg / tmp_0
    return tmp_1

def replacement_args(arg):
    """
    Extract argument for the replacement function
    """
    return (arg,)

def _next_power_of_2(n):
    """Return the next power of 2 greater than or equal to n"""
    return 1 if n == 0 else 2**((n - 1).bit_length())

@triton.jit
def l2_normalization_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Optimized Triton kernel for L2 normalization: y = x / ||x||_2
    where ||x||_2 is computed row-wise (dim=-1, keepdim=True)
    Matching PyTorch's behavior exactly for correctness
    """
    # Program identifiers for 2D grid
    m = tl.program_id(0)
    
    # Row bounds for this program
    row_start = m * BLOCK_SIZE_M
    
    # Exit if this program doesn't have any work to do
    if row_start >= n_rows:
        return
    
    # Initialize sum of squares for this row - use float32 for precision
    row_sum_sq = tl.zeros([], dtype=tl.float32)
    
    # Compute sum of squares for the entire row
    # Process columns in blocks
    col_offset = 0
    while col_offset < n_cols:
        # Create mask for current block
        col_mask = col_offset + tl.arange(0, BLOCK_SIZE_N) < n_cols
        
        # Load data block
        data = tl.load(
            x_ptr + row_start * n_cols + col_offset + tl.arange(0, BLOCK_SIZE_N),
            mask=col_mask,
            other=0.0
        ).to(tl.float32)
        
        # Accumulate sum of squares
        row_sum_sq += tl.sum(data * data)
        
        # Move to next block
        col_offset += BLOCK_SIZE_N
    
    # Compute L2 norm exactly like PyTorch does
    # PyTorch uses sqrt(sum_of_squares) for L2 norm
    row_norm = tl.sqrt(row_sum_sq)
    
    # Handle zero vectors like PyTorch does - if norm is too small, set to 1
    # This prevents division by zero while maintaining numerical stability
    row_norm = tl.maximum(row_norm, 1.0)
    
    # Now normalize each element in the row
    col_offset = 0
    while col_offset < n_cols:
        # Create mask for current block
        col_mask = col_offset + tl.arange(0, BLOCK_SIZE_N) < n_cols
        
        # Load input data block
        data = tl.load(
            x_ptr + row_start * n_cols + col_offset + tl.arange(0, BLOCK_SIZE_N),
            mask=col_mask,
            other=0.0
        ).to(tl.float32)
        
        # Compute normalized values
        normalized_vals = data / row_norm
        
        # Store results as bfloat16 to match input precision
        tl.store(
            out_ptr + row_start * n_cols + col_offset + tl.arange(0, BLOCK_SIZE_N),
            normalized_vals.to(tl.bfloat16),
            mask=col_mask
        )
        
        # Move to next block
        col_offset += BLOCK_SIZE_N

@torch.fx.wrap
def fused_l2_normalization(arg):
    """
    Wrapper function that performs L2 normalization using optimized Triton kernel
    Returns normalized tensor
    """
    n_rows, n_cols = arg.shape
    
    # Choose optimal block sizes based on tensor size
    if n_cols <= 512:
        BLOCK_SIZE_N = 256
    elif n_cols <= 1024:
        BLOCK_SIZE_N = 512
    else:
        BLOCK_SIZE_N = 1024
    
    BLOCK_SIZE_M = 64  # Process more rows per program for better occupancy
    
    # Calculate grid dimensions
    grid_m = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    
    # Allocate output for normalized tensor
    normalized_arg = torch.empty_like(arg)
    
    # Launch Triton kernel for L2 normalization (fused norm + division)
    l2_normalization_kernel[(grid_m,)](
        x_ptr=arg,
        out_ptr=normalized_arg,
        n_rows=n_rows,
        n_cols=n_cols,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    return normalized_arg

def replacement_func():
    """
    Returns the fused function that performs L2 normalization
    """
    return fused_l2_normalization