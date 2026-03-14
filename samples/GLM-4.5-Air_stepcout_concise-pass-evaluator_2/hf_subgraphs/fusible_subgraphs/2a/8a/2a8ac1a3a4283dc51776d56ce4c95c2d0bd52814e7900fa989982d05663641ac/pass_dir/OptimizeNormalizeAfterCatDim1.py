import torch
import triton
import triton.language as tl

def pattern(in_0):
    """
    Pattern matching for concatenation followed by L2 normalization.
    Note: torch.cat([in_0], 1) is a no-op that essentially returns in_0 unchanged.
    """
    tmp_0 = torch.cat([in_0], 1)  # Redundant concatenation
    tmp_1 = torch.nn.functional.normalize(tmp_0, p=2, dim=1)
    return tmp_1

def replacement_args(in_0):
    """
    Extract arguments needed for the normalized kernel.
    Since the concatenation is redundant, we only need the original input.
    """
    return (in_0,)

@triton.jit
def l2_normalize_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    dim_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for L2 normalization along specified dimension.
    This kernel performs L2 normalization along dimension 1 by:
    1. Computing L2 norm along dim=1 for each batch
    2. Dividing each row by its L2 norm
    """
    batch_size = n_elements // dim_size
    
    # Each program handles one row of the batch
    batch_idx = tl.program_id(0)
    row_idx = tl.program_id(1)
    
    # Global position for this row
    row_start = batch_idx * dim_size
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < row_start + dim_size
    
    # Load the row
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute sum of squares for L2 norm
    sum_squares = tl.sum(x * x)
    
    # Compute L2 norm (add small epsilon for numerical stability)
    norm = tl.sqrt(sum_squares + 1e-8)
    
    # Normalize the row
    x_normalized = x / norm
    
    # Store the normalized row
    tl.store(out_ptr + offsets, x_normalized, mask=mask)

@torch.fx.wrap
def optimized_l2_normalize(x):
    """
    Optimized L2 normalization function using Triton.
    This replaces both the redundant concatenation and the original normalize call.
    """
    batch_size, dim_size = x.shape
    
    # Use optimal block size based on input dimension
    BLOCK_SIZE = 1024
    if dim_size < 1024:
        BLOCK_SIZE = triton.next_power_of_2(dim_size)
    
    num_batches = batch_size
    num_rows_per_batch = 1
    
    num_programs_batch = (num_batches + 31) // 32
    num_programs_row = max(1, (dim_size + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    out = torch.empty_like(x)
    
    # Launch kernel with 2D grid: [num_batches, num_programs_per_batch]
    l2_normalize_kernel[(num_programs_batch, num_programs_row)](
        x_ptr=x,
        out_ptr=out,
        n_elements=x.numel(),
        dim_size=dim_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    """
    Return the optimized normalization function.
    """
    return optimized_l2_normalize