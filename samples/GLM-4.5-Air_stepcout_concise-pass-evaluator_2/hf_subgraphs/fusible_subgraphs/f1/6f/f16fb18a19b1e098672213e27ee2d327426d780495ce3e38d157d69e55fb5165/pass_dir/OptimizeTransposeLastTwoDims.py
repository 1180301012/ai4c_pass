import torch
import triton
import triton.language as tl

def tensor_transpose_last_two_dims(x):
    return x.transpose(-2, -1)

def replacement_args(x):
    return (x,)

@triton.jit
def transpose_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Each program handles one tile of the transpose
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)
    
    # Calculate tile ranges
    row_start = row_block * BLOCK_SIZE_M
    col_start = col_block * BLOCK_SIZE_N
    
    # Create column and row offsets within the tile
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_M)
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks to handle boundaries
    row_mask = row_offsets < n_rows
    col_mask = col_offsets < n_cols
    
    # Load input tile (row-major)
    x_tile = tl.load(x_ptr + (row_offsets[:, None] * n_cols + col_offsets[None, :]), 
                    mask=row_mask[:, None] & col_mask[None, :], other=0.0)
    
    # Transpose tile
    out_tile = x_tile.transpose(0, 1)
    
    # Store output tile (column-major becomes row-major after transpose)
    out_offset = (col_offsets[:, None] * n_rows + row_offsets[None, :])
    tl.store(out_ptr + out_offset, out_tile, mask=col_mask[:, None] & row_mask[None, :])

@torch.fx.wrap
def optimized_transpose(x):
    if x.dim() < 2:
        return x.clone()
        
    # Get the last two dimensions
    n_rows, n_cols = x.shape[-2:]
    
    # Allocate output with same shape
    out = torch.empty_like(x)
    
    # For tensors with more than 2 dimensions, process each slice
    if x.dim() > 2:
        leading_dims = x.shape[:-2]
        total_elements = 1
        for dim in leading_dims:
            total_elements *= dim
        
        for i in range(total_elements):
            # Simple index calculation for leading dimensions
            indices = []
            temp = i
            for dim in reversed(leading_dims):
                indices.insert(0, temp % dim)
                temp //= dim
            
            # Get slice for this batch element
            x_slice = x[tuple(indices) + (slice(None), slice(None))]
            out_slice = out[tuple(indices) + (slice(None), slice(None))]
            
            # Transpose this slice
            _transpose_slice(x_slice, out_slice, n_rows, n_cols)
    else:
        _transpose_slice(x, out, n_rows, n_cols)
    
    return out

def _transpose_slice(x, out, n_rows, n_cols):
    # Set up block sizes based on matrix dimensions
    if n_rows >= 512 and n_cols >= 512:
        BLOCK_SIZE_M = 32
        BLOCK_SIZE_N = 32
    elif n_rows >= 256 and n_cols >= 256:
        BLOCK_SIZE_M = 64 
        BLOCK_SIZE_N = 64
    else:
        BLOCK_SIZE_M = min(128, n_rows)
        BLOCK_SIZE_N = min(128, n_cols)
    
    # Calculate grid dimensions
    grid_m = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (n_cols + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    transpose_kernel[(grid_m, grid_n)](
        x_ptr=x,
        out_ptr=out,
        n_rows=n_rows,
        n_cols=n_cols,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )

def replacement_func():
    return optimized_transpose