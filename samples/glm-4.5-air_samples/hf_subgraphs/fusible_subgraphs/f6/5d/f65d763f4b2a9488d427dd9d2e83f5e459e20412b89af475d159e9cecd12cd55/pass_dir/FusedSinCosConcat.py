import torch
import triton
import triton.language as tl

def pattern(x):
    return torch.cat((x.cos(), x.sin()), dim=-1)

def replacement_args(x):
    return (x,)



@triton.jit
def fused_sincos_kernel(
    x_ptr,
    out_ptr,
    rows,
    cols,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Each program handles a block of the 2D tensor
    row_idx = tl.program_id(0)
    col_offset = tl.program_id(1) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = col_offset < cols
    
    # Load input data for this row
    x_ptr_row = x_ptr + row_idx * cols
    x = tl.load(x_ptr_row + col_offset, mask=mask, other=0.0)
    
    # Compute both sin and cos simultaneously
    cos_x = tl.cos(x)
    sin_x = tl.sin(x)
    
    # Determine output pointers for this row
    # Output shape: [rows, cols*2] where first cols are cos, next cols are sin
    output_row_ptr = out_ptr + row_idx * (cols * 2)
    
    # Store cos in first half, sin in second half
    tl.store(output_row_ptr + col_offset, cos_x, mask=mask)
    tl.store(output_row_ptr + cols + col_offset, sin_x, mask=mask)

@torch.fx.wrap
def fused_sincos_optimized(x):
    rows, cols = x.shape
    
    # Allocate output tensor with correct 2D shape [rows, cols*2]
    output = torch.empty((rows, cols * 2), dtype=x.dtype, device=x.device)
    
    # Optimized block sizes for our specific tensor [64, 64]
    BLOCK_SIZE_M = 1  # Process one row at a time
    BLOCK_SIZE_N = cols  # Process entire row width (all columns at once)
    
    # Calculate grid
    num_rows = (rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_cols = (cols + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch kernel
    fused_sincos_kernel[(num_rows, num_cols)](
        x,
        output,
        rows,
        cols,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
    )
    
    return output

def replacement_func():
    return fused_sincos_optimized