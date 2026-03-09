import torch
import triton
import triton.language as tl

def pattern(x, weight, bias, other_weight, other_bias):
    # Pattern: view operation followed by unsqueeze 
    tmp_6 = x.view(-1, 256)
    tmp_13 = tmp_6.unsqueeze(-2)
    return tmp_13

def replacement_args(x, weight, bias, other_weight, other_bias):
    return (x,)

@triton.jit
def reshape_unsqueeze_kernel(
    x_ptr, out_ptr,
    n_total_rows, n_cols,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    row_idx = pid_m * BLOCK_SIZE_M + pid_n
    
    if row_idx >= n_total_rows:
        return
    
    # Load input row
    x_row = tl.load(x_ptr + row_idx * n_cols)
    
    # Store output with extra dimension (unsqueeze(-2))
    out_offset = row_idx * BLOCK_SIZE_N * n_cols
    for k in range(n_cols):
        tl.store(out_ptr + out_offset + k, x_row[k])

@torch.fx.wrap
def optimized_reshape_unsqueeze(x):
    n_total_rows, n_cols = x.shape
    target_cols = 256
    
    # For this specific case, we know we want unsqueeze(-2)
    # which means adding a dimension at position -2 (second from last)
    # [N, 256] → [N, 1, 256]
    output_shape = (n_total_rows, 1, target_cols)
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    # Use Triton to perform this efficiently
    BLOCK_SIZE_M = 32  # Number of rows per program
    BLOCK_SIZE_N = 1   # Process one row at a time per inner dimension
    
    grid = (
        triton.cdiv(n_total_rows, BLOCK_SIZE_M),
        target_cols
    )
    
    # Simple kernel that just reshapes and adds the singleton dimension
    @triton.jit
    def reshape_kernel(x_ptr, out_ptr, n_rows, n_cols, BLOCK_SIZE_M: tl.constexpr):
        pid_m = tl.program_id(0)
        row_start = pid_m * BLOCK_SIZE_M
        row_end = min(row_start + BLOCK_SIZE_M, n_rows)
        
        for row_idx in range(row_start, row_end):
            # Load original row
            x_row = tl.load(x_ptr + row_idx * n_cols)
            
            # Store with new shape [N, 1, C] 
            # The middle dimension of 1 is handled at tensor level
            out_offset = row_idx * n_cols
            for k in range(n_cols):
                tl.store(out_ptr + out_offset + k, x_row[k])
    
    reshape_kernel[grid](x, out.view(-1), n_total_rows, target_cols, BLOCK_SIZE_M)
    
    return out

def replacement_func():
    return optimized_reshape_unsqueeze