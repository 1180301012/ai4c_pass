import torch
import triton
import triton.language as tl

def pattern(tmp_1):
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1, dtype=torch.float32)
    tmp_3 = tmp_2.to(torch.float32)
    tmp_4 = torch.nn.functional.dropout(tmp_3, p=0.0, training=False)
    return tmp_4

def replacement_args(tmp_1):
    return (tmp_1, )

@triton.jit
def softmax_kernel(
    x_ptr,
    y_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1)

    # Calculate row start
    row_start = row * n_cols
    
    # Load row data in tiles
    offsets = col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    x = tl.load(x_ptr + row_start + offsets, mask=mask)

    # Compute row max (using shared memory for efficiency)
    row_max = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    row_max = tl.max(x, axis=0)
    
    # Compute exponential with shift
    exp_x = tl.exp(x - row_max)
    
    # Compute row sum
    row_sum = tl.sum(exp_x)
    
    # Normalize
    y = exp_x / row_sum
    
    # Store result
    tl.store(y_ptr + row_start + offsets, y, mask=mask)

@torch.fx.wrap
def softmax_triton(x):
    n_rows, n_cols = x.shape
    grid = (n_rows, (n_cols + 128 - 1) // 128)
    y = torch.empty_like(x)
    softmax_kernel[grid](
        x_ptr=x,
        y_ptr=y,
        n_rows=n_rows,
        n_cols=n_cols,
        BLOCK_SIZE=128,
    )
    return y

def replacement_func():
    return softmax_triton