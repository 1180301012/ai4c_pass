import torch
import triton
import triton.language as tl

def pattern(x):
    t1 = x / 16.0
    t2 = t1 / 0.05
    return t2.softmax(dim=-1)

def replacement_args(x):
    return (x,)

@triton.jit
def softmax_kernel(X_ptr, Y_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    block_id = tl.program_id(1)
    row_start = row_id * n_cols
    start_idx = row_start + block_id * BLOCK_SIZE
    x = tl.load(X_ptr + start_idx, num_elements=BLOCK_SIZE, mask=start_idx + tl.arange(0, BLOCK_SIZE) < row_start + n_cols)
    row_max = tl.max(x)
    row_max = tl.broadcast(row_max, [1, BLOCK_SIZE])
    x = x - row_max
    x = tl.exp(x)
    row_sum = tl.sum(x)
    x = x / row_sum
    tl.store(Y_ptr + start_idx, x, mask=start_idx + tl.arange(0, BLOCK_SIZE) < row_start + n_cols)

@torch.fx.wrap
def optimized_softmax(x):
    orig_shape = x.shape
    x_flat = x.view(-1, orig_shape[-1])
    n_cols = orig_shape[-1]
    num_rows = x_flat.shape[0]
    BLOCK_SIZE = 512
    num_blocks = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
    y_flat = torch.empty_like(x_flat)
    softmax_kernel[(num_rows, num_blocks)](x_flat, y_flat, n_cols, BLOCK_SIZE)
    return y_flat.view(orig_shape)

def replacement_func():
    return optimized_softmax