import torch
import triton
import triton.language as tl

def pattern(x):
    """Pattern: scalar multiplication followed by softmax on last dimension"""
    scaled = 0.0625 * x
    softmax_out = torch.nn.functional.softmax(scaled, dim=-1)
    return softmax_out

def replacement_args(x):
    """Extract arguments for the fused scale-softmax operation"""
    return (x,)

@triton.jit
def fused_scale_softmax_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    n_batches,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Fused kernel that scales and applies softmax in one pass"""
    # Each program handles one row per block
    batch_id = tl.program_id(0)
    row_id = tl.program_id(1)
    
    batch_offset = batch_id * n_rows * n_cols
    row_offset = row_id * n_cols
    
    # Load the row
    x_ptr_row = x_ptr + batch_offset + row_offset
    x = tl.load(x_ptr_row, mask=tl.arange(0, BLOCK_SIZE_N) < n_cols, other=float('-inf'))
    
    # Apply scale (multiply by 0.0625 = 1/16)
    scaled_x = x * 0.0625
    
    # Compute max for numerical stability
    max_val = tl.max(scaled_x, axis=0)
    scaled_x = scaled_x - max_val
    
    # Compute softmax
    exp_x = tl.exp(scaled_x)
    sum_exp = tl.sum(exp_x, axis=0)
    softmax_out = exp_x / sum_exp
    
    # Store result
    out_ptr_row = out_ptr + batch_offset + row_offset
    tl.store(out_ptr_row, softmax_out, mask=tl.arange(0, BLOCK_SIZE_N) < n_cols)

@torch.fx.wrap
def fused_scale_softmax(x):
    """Wrapper function for fused scale-softmax kernel"""
    batch_size, n_rows, n_cols = x.shape
    
    # Optimal block sizes for GPU
    BLOCK_SIZE_M = 64  # Rows per block
    BLOCK_SIZE_N = 256  # Columns per block
    
    # Calculate grid dimensions
    n_batches = batch_size
    n_rows_total = n_rows
    n_cols_total = n_cols
    
    # Create output tensor
    output = torch.empty_like(x)
    
    # Launch kernel
    grid = (
        n_batches,
        (n_rows_total + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,
    )
    
    fused_scale_softmax_kernel[grid](
        x_ptr=x,
        out_ptr=output,
        n_rows=n_rows_total,
        n_cols=n_cols_total,
        n_batches=n_batches,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return output

def replacement_func():
    """Return the fused scale-softmax function"""
    return fused_scale_softmax