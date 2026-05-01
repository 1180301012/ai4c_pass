import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)
    col_start = tl.program_id(1) * BLOCK_SIZE
    
    # First pass: compute row max
    row_max = -float('inf')
    for i in range(0, n_cols, BLOCK_SIZE):
        data = tl.load(input_ptr + row * n_cols + i, 
                     mask=(i + tl.arange(0, BLOCK_SIZE)) < n_cols, 
                     other=-float('inf'))
        row_max = tl.maximum(row_max, tl.max(data))
    
    # Second pass: compute exp and sum
    row_sum = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        data = tl.load(input_ptr + row * n_cols + i, 
                     mask=(i + tl.arange(0, BLOCK_SIZE)) < n_cols, 
                     other=0.0)
        data = data - row_max
        exp_data = tl.exp(data)
        row_sum += tl.sum(exp_data)
        tl.store(output_ptr + row * n_cols + i, exp_data, 
                mask=(i + tl.arange(0, BLOCK_SIZE)) < n_cols)
    
    # Third pass: normalize
    for i in range(0, n_cols, BLOCK_SIZE):
        exp_data = tl.load(output_ptr + row * n_cols + i, 
                        mask=(i + tl.arange(0, BLOCK_SIZE)) < n_cols, 
                        other=0.0)
        exp_data = exp_data / row_sum
        tl.store(output_ptr + row * n_cols + i, exp_data, 
                mask=(i + tl.arange(0, BLOCK_SIZE)) < n_cols)

@torch.fx.wrap
def triton_softmax(x):
    batch, seq1, seq2 = x.shape
    x_flat = x.reshape(batch * seq1, seq2)
    n_rows, n_cols = x_flat.shape
    out = torch.empty_like(x_flat)
    BLOCK_SIZE = 1024
    grid = (n_rows, (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE)
    softmax_kernel[grid](x_flat, out, n_rows, n_cols, BLOCK_SIZE)
    return out.reshape(batch, seq1, seq2)

def pattern(a):
    return a.softmax(dim=-1)

def replacement_args(a):
    return (a,)

def replacement_func():
    return triton_softmax