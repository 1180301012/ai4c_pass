import torch
import triton
import triton.language as tl

def pattern(in_5, in_1, in_0):
    return torch.nn.functional.linear(in_5, in_1, in_0)

def replacement_args(in_5, in_1, in_0):
    return (in_5, in_1, in_0)

@triton.jit
def simple_linear_kernel(
    x_ptr,
    w_ptr, 
    b_ptr,
    out_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    row_start = pid_m * BLOCK_SIZE_M
    row_end = min((pid_m + 1) * BLOCK_SIZE_M, n_rows)
    rows = row_end - row_start
    
    col_start = pid_n * BLOCK_SIZE_K  
    col_end = min((pid_n + 1) * BLOCK_SIZE_K, n_cols)
    cols = col_end - col_start
    
    # Initialize accumulator to zero
    acc = 0.0
    
    # Process each element in the output block
    for i in range(rows):
        for j in range(cols):
            acc = 0.0
            
            # Matrix multiplication for element (i,j)
            for k in range(256):
                # Load input element
                x_offset = (row_start + i) * 256 + k
                x_val = tl.load(x_ptr + x_offset, mask=x_offset < n_rows * 256, other=0.0)
                
                # Load weight element  
                w_offset = k * n_cols + (col_start + j)
                w_val = tl.load(w_ptr + w_offset, mask=w_offset < 256 * n_cols, other=0.0)
                
                # Multiply and accumulate
                acc += x_val * w_val
            
            # Add bias
            b_offset = col_start + j
            b_val = tl.load(b_ptr + b_offset, mask=b_offset < n_cols, other=0.0)
            acc += b_val
            
            # Store result
            out_offset = (row_start + i) * n_cols + (col_start + j)
            tl.store(out_ptr + out_offset, acc)

@torch.fx.wrap
def optimized_linear(in_5, in_1, in_0):
    n_rows = in_5.shape[0]
    n_cols = in_1.shape[0]
    
    out = torch.empty((n_rows, n_cols), dtype=in_5.dtype, device=in_5.device)
    
    grid = lambda meta: (
        (n_rows + meta['BLOCK_SIZE_M'] - 1) // meta['BLOCK_SIZE_M'],
        (n_cols + meta['BLOCK_SIZE_K'] - 1) // meta['BLOCK_SIZE_K'],
    )
    
    simple_linear_kernel[grid](
        in_5, in_1, in_0, out, n_rows, n_cols,
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_K=32,
    )
    
    return out

def replacement_func():
    return optimized_linear