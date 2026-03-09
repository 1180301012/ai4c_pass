import torch
import triton
import triton.language as tl

def pattern(x, weight, bias, other_weight, other_bias):
    # Pattern: linear transformation followed by splitting output into two halves
    # This matches: tmp_4 = linear(in_5, tmp_1, tmp_0)
    # tmp_5 = tmp_4[:, :256], tmp_6 = tmp_5.view(-1, 256) 
    # tmp_7 = tmp_4[:, -256:], tmp_8 = tmp_7.view(-1, 256)
    tmp_4 = torch.nn.functional.linear(x, weight, bias)
    tmp_5 = tmp_4[:, :256]  # first 256 columns
    tmp_6 = tmp_5.view(-1, 256)
    tmp_7 = tmp_4[:, -256:]  # last 256 columns
    tmp_8 = tmp_7.view(-1, 256)
    return tmp_6, tmp_8

def replacement_args(x, weight, bias, other_weight, other_bias):
    return (x, weight, bias)

@triton.jit
def linear_split_kernel(
    x_ptr, weight_ptr, bias_ptr,
    out1_ptr, out2_ptr,
    n_rows, n_cols1, n_cols2,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    # Program ID for rows
    pid_m = tl.program_id(0)
    # Start offset for this program's rows
    row_start = pid_m * BLOCK_SIZE_M
    row_end = min(row_start + BLOCK_SIZE_M, n_rows)
    
    # If this program doesn't have any work to do, exit
    if row_start >= n_rows:
        return
        
    # Load bias
    bias = tl.load(bias_ptr)
    
    # Process one row at a time
    for row_idx in range(row_start, row_end):
        # Load input row - need to handle the case where x might not be contiguous
        if n_rows == 1:
            x_row = tl.load(x_ptr)
        else:
            x_row = tl.load(x_ptr + row_idx * n_cols1)
        
        # Compute first output half
        acc1 = tl.zeros((n_cols1,), dtype=tl.float32)
        for k in range(n_cols1):
            acc1[k] = bias[k]
            for j in range(n_cols2):
                acc1[k] += x_row[j] * weight_ptr[j * n_cols2 + k]
        
        # Compute second output half  
        acc2 = tl.zeros((n_cols2,), dtype=tl.float32)
        for k in range(n_cols2):
            acc2[k] += bias[n_cols1 + k]
            for j in range(n_cols2):
                acc2[k] += x_row[j] * weight_ptr[j * n_cols2 + n_cols1 + k]
        
        # Store results
        tl.store(out1_ptr + row_idx * n_cols1, acc1)
        tl.store(out2_ptr + row_idx * n_cols2, acc2)

@torch.fx.wrap
def fused_linear_split(x, weight, bias):
    n_rows, n_cols = x.shape
    total_out_cols = weight.shape[0]
    split_cols = total_out_cols // 2
    
    out1 = torch.empty((n_rows, split_cols), dtype=x.dtype, device=x.device)
    out2 = torch.empty((n_rows, split_cols), dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 256
    
    grid = (triton.cdiv(n_rows, BLOCK_SIZE_M),)
    
    linear_split_kernel[grid](
        x, weight, bias,
        out1, out2,
        n_rows, split_cols, split_cols,
        BLOCK_SIZE_M, BLOCK_SIZE_N
    )
    
    return out1.view(-1, split_cols), out2.view(-1, split_cols)

def replacement_func():
    return fused_linear_split