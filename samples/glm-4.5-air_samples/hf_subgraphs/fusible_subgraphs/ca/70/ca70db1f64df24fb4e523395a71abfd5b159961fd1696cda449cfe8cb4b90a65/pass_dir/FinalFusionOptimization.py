import torch
import triton
import triton.language as tl

def pattern(x, weight, bias):
    # Simplified pattern: linear transformation followed by explicit splitting
    # This matches the exact computation from the original computation graph
    tmp_4 = torch.nn.functional.linear(x, weight, bias)
    tmp_5 = tmp_4[:, :256]  # First 256 columns
    tmp_6 = tmp_5.view(-1, 256)
    tmp_7 = tmp_4[:, -256:]  # Last 256 columns  
    tmp_8 = tmp_7.view(-1, 256)
    return tmp_6, tmp_8

def replacement_args(x, weight, bias):
    return (x, weight, bias)

@triton.jit
def fused_linear_split_kernel(
    x_ptr, weight_ptr, bias_ptr, 
    out1_ptr, out2_ptr,
    n_rows, n_cols_in, total_cols_out,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    row_idx = pid_m
    col_start = pid_n * BLOCK_SIZE_N
    
    if row_idx >= n_rows or col_start >= total_cols_out:
        return
    
    # Process both output halves together for better memory efficiency
    cols_per_half = total_cols_out // 2
    
    for k in range(BLOCK_SIZE_N):
        if col_start + k < total_cols_out:
            offset = row_idx * total_cols_out + col_start + k
            
            # Load bias element
            if k < cols_per_half:
                # First half: bias from first part
                out_val = tl.load(bias_ptr + k)
            else:
                # Second half: bias from second part  
                out_val = tl.load(bias_ptr + k)
            
            # Matrix multiplication
            for j in range(n_cols_in):
                x_val = tl.load(x_ptr + row_idx * n_cols_in + j)
                weight_val = tl.load(weight_ptr + j * total_cols_out + col_start + k)
                out_val += x_val * weight_val
            
            # Store to appropriate output
            if k < cols_per_half:
                out1_offset = row_idx * cols_per_half + k
                tl.store(out1_ptr + out1_offset, out_val)
            else:
                out2_offset = row_idx * cols_per_half + (k - cols_per_half)
                tl.store(out2_ptr + out2_offset, out_val)

@torch.fx.wrap
def optimized_fused_linear_split(x, weight, bias):
    n_rows, n_cols_in = x.shape
    total_cols_out = weight.shape[0]
    
    # Output tensors
    cols_per_half = total_cols_out // 2
    out1 = torch.empty((n_rows, cols_per_half), dtype=x.dtype, device=x.device)
    out2 = torch.empty((n_rows, cols_per_half), dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 256
    
    grid = (
        triton.cdiv(n_rows, BLOCK_SIZE_M),
        triton.cdiv(total_cols_out, BLOCK_SIZE_N)
    )
    
    fused_linear_split_kernel[grid](
        x, weight, bias,
        out1, out2,
        n_rows, n_cols_in, total_cols_out,
        BLOCK_SIZE_M, BLOCK_SIZE_N
    )
    
    # Reshape outputs to match original pattern
    return out1.view(-1, cols_per_half), out2.view(-1, cols_per_half)

def replacement_func():
    return optimized_fused_linear_split