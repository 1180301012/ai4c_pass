import torch
import triton
import triton.language as tl

def pattern(linear):
    tmp_3 = torch.reshape(linear, [-1, 9, 1])
    tmp_4 = torch.softmax(tmp_3, dim = 1)
    return tmp_4

def replacement_args(linear):
    return (linear,)

@triton.jit
def softmax_kernel(x_ptr, out_ptr, n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    if row >= n_rows:
        return
    row_start = row * n_cols
    row_max = -1e30
    for i in range(n_cols):
        val = tl.load(x_ptr + row_start + i)
        row_max = tl.maximum(row_max, val)
    row_sum = 0.0
    for i in range(n_cols):
        val = tl.load(x_ptr + row_start + i)
        exp_val = tl.exp(val - row_max)
        row_sum += exp_val
    for i in range(n_cols):
        val = tl.load(x_ptr + row_start + i)
        exp_val = tl.exp(val - row_max)
        softmax_val = exp_val / row_sum
        tl.store(out_ptr + row_start + i, softmax_val)

@torch.fx.wrap
def softmax_wrapper(linear):
    n_rows = linear.numel() // 9
    reshaped_flat = torch.empty(n_rows, 9, dtype=linear.dtype)
    for i in range(n_rows):
        for j in range(9):
            reshaped_flat[i, j] = linear[i * 9 + j]
    out_flat = torch.empty_like(reshaped_flat)
    softmax_kernel[(n_rows,)](
        x_ptr=reshaped_flat,
        out_ptr=out_flat,
        n_rows=n_rows,
        n_cols=9,
        BLOCK_SIZE=8
    )
    out_3d = torch.empty(n_rows, 9, 1, dtype=linear.dtype)
    for i in range(n_rows):
        for j in range(9):
            out_3d[i, j, 0] = out_flat[i, j]
    return out_3d

def replacement_func():
    return softmax_wrapper