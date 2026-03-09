import torch
import triton
import triton.language as tl

def pattern(x, weight, bias):
    # Simple pattern: just linear transformation
    out = torch.nn.functional.linear(x, weight, bias)
    return out

def replacement_args(x, weight, bias):
    return (x, weight, bias)

@triton.jit
def simple_linear_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    n_rows, n_cols_in, n_cols_out,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    pid_m = tl.program_id(0)  # Each program handles one row
    pid_n = tl.program_id(1)  # Each program handles a block of output columns
    
    # Row index
    row_idx = pid_m
    if row_idx >= n_rows:
        return
    
    # Output column range
    col_start = pid_n * BLOCK_SIZE_N
    
    # Load bias for all elements in the column range
    for k in range(BLOCK_SIZE_N):
        if col_start + k < n_cols_out:
            offset = row_idx * n_cols_out + col_start + k
            # Initialize with bias
            out_val = tl.load(bias_ptr + col_start + k)
            
            # Matrix multiplication
            for j in range(n_cols_in):
                # Load input element
                x_val = tl.load(x_ptr + row_idx * n_cols_in + j, mask=j < n_cols_in, other=0.0)
                
                # Load weight element
                weight_val = tl.load(weight_ptr + j * n_cols_out + col_start + k)
                
                # Accumulate
                out_val += x_val * weight_val
            
            # Store result
            tl.store(out_ptr + offset, out_val)

@torch.fx.wrap
def optimized_linear(x, weight, bias):
    # Handle different input dimensions - support both 2D and 3D inputs
    if x.dim() == 2:
        n_rows, n_cols_in = x.shape
    elif x.dim() == 3:
        n_rows, seq_len, n_cols_in = x.shape
    else:
        raise ValueError(f"Unsupported input dimension: {x.dim()}")
    
    n_cols_out = weight.shape[0]
    
    if x.dim() == 2:
        out = torch.empty((n_rows, n_cols_out), dtype=x.dtype, device=x.device)
        target_shape = (n_rows, n_cols_out)
    else:  # 3D
        out = torch.empty((n_rows, seq_len, n_cols_out), dtype=x.dtype, device=x.device)
        target_shape = (n_rows, seq_len, n_cols_out)
    
    BLOCK_SIZE_M = 32  # Each program processes one row
    BLOCK_SIZE_N = 256  # Each program processes 256 output columns
    
    # Create grid
    grid = (
        triton.cdiv(n_rows, BLOCK_SIZE_M),
        triton.cdiv(n_cols_out, BLOCK_SIZE_N)
    )
    
    # Reshape input to 2D for consistent processing
    if x.dim() == 3:
        x_reshaped = x.reshape(-1, n_cols_in)
        out_reshaped = out.reshape(-1, n_cols_out)
        n_total_rows = n_rows * seq_len
    else:
        x_reshaped = x
        out_reshaped = out
        n_total_rows = n_rows
    
    simple_linear_kernel[grid](
        x_reshaped, weight, bias, out_reshaped,
        n_total_rows, n_cols_in, n_cols_out,
        BLOCK_SIZE_M, BLOCK_SIZE_N
    )
    
    return out

def replacement_func():
    return optimized_linear