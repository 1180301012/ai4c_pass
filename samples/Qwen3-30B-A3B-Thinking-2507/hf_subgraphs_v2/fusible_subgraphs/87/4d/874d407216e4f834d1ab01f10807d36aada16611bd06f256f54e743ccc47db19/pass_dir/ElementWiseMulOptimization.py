import torch
import triton
import triton.language as tl

@triton.jit
def element_wise_mul_kernel(x_ptr, y_ptr, out_ptr, n_rows, n_cols, BLOCK_ROWS: tl.constexpr):
    block_id = tl.program_id(0)
    start_row = block_id * BLOCK_ROWS
    row = start_row + tl.arange(0, BLOCK_ROWS)
    row_mask = row < n_rows

    x = tl.load(x_ptr + row * 1, mask=row_mask, other=0.0)
    y = tl.load(y_ptr + row * n_cols, mask=row_mask[:, None], other=0.0)

    out = x[:, None] * y
    tl.store(out_ptr + row * n_cols, out, mask=row_mask[:, None])

@torch.fx.wrap
def optimized_mul(in_1, in_2):
    n_rows = in_1.shape[0]
    n_cols = in_2.shape[1]
    out = torch.empty_like(in_2)
    BLOCK_ROWS = 128
    grid = (max(1, (n_rows + BLOCK_ROWS - 1) // BLOCK_ROWS),)
    
    x = in_1.view(-1, 1).contiguous()
    y = in_2.contiguous()
    
    element_wise_mul_kernel[grid](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_rows=n_rows,
        n_cols=n_cols,
        BLOCK_ROWS=BLOCK_ROWS
    )
    return out

def kernel_wrapper(in_0, in_1, in_2):
    tmp_0 = in_1.view(-1, 1)
    tmp_1 = optimized_mul(in_1, in_2)
    tmp_2 = in_0.view(-1, 1)
    tmp_3 = tmp_2.expand_as(tmp_1)
    tmp_4 = tmp_1.new_zeros((1000, 16))
    return (tmp_3, tmp_4, tmp_1)

def pattern(in_0, in_1, in_2):
    tmp_0 = in_1.view(-1, 1)
    tmp_1 = tmp_0 * in_2
    tmp_2 = in_0.view(-1, 1)
    tmp_3 = tmp_2.expand_as(tmp_1)
    tmp_4 = tmp_1.new_zeros((1000, 16))
    return (tmp_3, tmp_4, tmp_1)

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

def replacement_func():
    return kernel_wrapper