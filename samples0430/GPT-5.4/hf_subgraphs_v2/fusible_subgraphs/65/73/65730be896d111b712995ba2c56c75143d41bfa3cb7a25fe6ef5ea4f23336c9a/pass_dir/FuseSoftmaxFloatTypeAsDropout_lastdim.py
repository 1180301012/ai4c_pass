import torch
import triton
import triton.language as tl


def pattern(in_2):
    tmp_1 = in_2.float()
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_3 = tmp_2.type_as(in_2)
    tmp_4 = torch.nn.functional.dropout(tmp_3, p=0.1, training=False)
    return tmp_4



def replacement_args(in_2):
    return (in_2,)


@triton.jit
def _softmax_lastdim_kernel(
    x_ptr,
    out_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    row_start = row_idx * n_cols
    mask = offs < n_cols

    x = tl.load(x_ptr + row_start + offs, mask=mask, other=-float('inf'))
    x = x.to(tl.float32)
    x_max = tl.max(x, axis=0)
    x = x - x_max
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    y = num / den
    tl.store(out_ptr + row_start + offs, y, mask=mask)


@torch.fx.wrap
def fused_softmax_float_typeas_dropout_lastdim(x):
    out = torch.empty_like(x)

    n_cols = x.shape[-1]
    if n_cols == 0:
        return out

    n_rows = x.numel() // n_cols
    block_size = triton.next_power_of_2(n_cols)
    if block_size > 1024:
        block_size = 1024

    if block_size <= 32:
        num_warps = 1
    elif block_size <= 128:
        num_warps = 2
    elif block_size <= 256:
        num_warps = 4
    else:
        num_warps = 8

    _softmax_lastdim_kernel[(n_rows,)](
        x,
        out,
        n_cols,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
        num_stages=1,
    )
    return out



def replacement_func():
    return fused_softmax_float_typeas_dropout_lastdim