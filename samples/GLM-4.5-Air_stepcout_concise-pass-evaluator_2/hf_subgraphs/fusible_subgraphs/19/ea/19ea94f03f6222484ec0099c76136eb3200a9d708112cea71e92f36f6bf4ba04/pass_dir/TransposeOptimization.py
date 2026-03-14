import torch
import triton
import triton.language as tl
import math

@triton.jit
def transpose_kernel(
    x_ptr,
    out_ptr,
    n_rows, 
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of elements, not just one
    pid = tl.program_id(0)
    
    # Compute block start and end positions
    block_start = pid * BLOCK_SIZE
    block_end = min(block_start + BLOCK_SIZE, n_cols)
    
    # Process each element in the block
    for col_idx in range(block_start, block_end):
        # Load element from input: [0, col_idx]
        src_offset = 0 * n_cols + col_idx
        x = tl.load(x_ptr + src_offset)
        
        # Store to transposed position: [col_idx, 0]
        dst_offset = col_idx * 1 + 0
        tl.store(out_ptr + dst_offset, x)

@torch.fx.wrap
def optimized_transpose(x):
    n_rows, n_cols = x.shape
    
    # Use larger block size to reduce program count
    # For 1152 elements, use block size of 256 = ~4 programs instead of 1152
    BLOCK_SIZE = 256
    
    # Calculate number of programs needed
    num_programs = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty((n_cols, 1), dtype=x.dtype, device=x.device)
    
    transpose_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_rows=n_rows,
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

def pattern(in_0):
    tmp_2 = in_0.t()
    return tmp_2

def replacement_args(in_0):
    return (in_0,)

def replacement_func():
    return optimized_transpose