import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    tmp_0 = in_1 + in_0
    tmp_1 = tmp_0.view(8, 300, 625)
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_3 = tmp_2.view(1, 8, 300, 625)
    tmp_4 = tmp_3.view(8, 300, 625)
    tmp_5 = torch.nn.functional.dropout(tmp_4, p=0.0, training=False)
    return (tmp_5, tmp_3)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def softmax_kernel(
    x_ptr,
    y_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    mask = (offset < n_elements)
    x = tl.load(x_ptr + offset, mask=mask, other=0.0)
    max_val = tl.max(x)
    y = tl.exp(x - max_val)
    sum_y = tl.sum(y)
    y = y / sum_y
    tl.store(y_ptr + offset, y, mask=mask)

@torch.fx.wrap
def kernel_wrapper(in_0, in_1):
    tmp_0 = in_1 + in_0
    tmp_1 = tmp_0.view(8, 300, 625)
    n_elements = tmp_1.numel()
    out = torch.empty_like(tmp_1)
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    softmax_kernel[(num_blocks,)](
        x_ptr=tmp_1,
        y_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    tmp_3 = out.view(1, 8, 300, 625)
    tmp_4 = tmp_3.view(8, 300, 625)
    return (tmp_4, tmp_3)

def replacement_func():
    return kernel_wrapper