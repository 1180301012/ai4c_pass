import torch
import triton
import triton.language as tl

def pattern(tmp_7, in_4, in_3):
    linear = torch.nn.functional.linear(tmp_7, in_4, in_3)
    tmp_9 = torch.tanh(linear)
    return tmp_9

def replacement_args(tmp_7, in_4, in_3):
    return (tmp_7, in_4, in_3)

@triton.jit
def linear_tanh_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE_K: tl.constexpr,
):
    program_id = tl.program_id(0)
    
    # For this pattern, n_rows is 1 (from slicing), so we process single row
    row_offset = 0
    row_mask = row_offset < n_rows
    
    # Load the single input row
    x = tl.load(x_ptr)
    
    # Load weight matrix in blocks
    col_offsets = tl.arange(0, BLOCK_SIZE_K)
    weight_ptrs = weight_ptr + col_offsets
    weight_cols = tl.load(weight_ptrs, mask=col_offsets < n_cols)
    
    # Process multiple columns in parallel using vectorized operations
    for k in range(0, n_cols, BLOCK_SIZE_K):
        # Load current weight block
        col_start = k
        col_end = min(k + BLOCK_SIZE_K, n_cols)
        col_mask = tl.arange(col_start, col_end) < n_cols
        
        weight_block = tl.load(weight_ptr + tl.arange(col_start, col_end), mask=col_mask)
        
        # Compute dot product for current output block
        out_block = tl.dot(x, weight_block, allow_tf32=False)
        
        # Load bias for corresponding columns
        bias_block = tl.load(bias_ptr + tl.arange(col_start, col_end), mask=col_mask)
        
        # Apply tanh activation
        out_block = tl.tanh(out_block + bias_block)
        
        # Store result
        out_ptrs = out_ptr + tl.arange(col_start, col_end)
        tl.store(out_ptrs, out_block, mask=col_mask)

@torch.fx.wrap
def fused_linear_tanh(x, weight, bias):
    n_rows, n_cols = x.shape
    
    out = torch.empty((n_rows, n_cols), dtype=x.dtype, device=x.device)
    
    # Since x has only 1 row (from slice operation), we use a single program
    linear_tanh_kernel[1, 1](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        n_rows=n_rows,
        n_cols=n_cols,
        BLOCK_SIZE_K=128  # Optimized vector size for 384 columns
    )
    
    return out

def replacement_func():
    return fused_linear_tanh