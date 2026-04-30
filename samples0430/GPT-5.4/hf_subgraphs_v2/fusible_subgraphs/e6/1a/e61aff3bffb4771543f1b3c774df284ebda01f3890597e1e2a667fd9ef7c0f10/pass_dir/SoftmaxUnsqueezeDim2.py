import torch
import triton
import triton.language as tl


def pattern(x):
    tmp_4 = torch.softmax(x, 2)
    tmp_5 = tmp_4.unsqueeze(-1)
    return tmp_5


def replacement_args(x):
    return (x,)


@triton.jit
def _softmax_unsqueeze_kernel(
    x_ptr,
    out_ptr,
    dim1,
    cols,
    x_stride0,
    x_stride1,
    x_stride2,
    out_stride0,
    out_stride1,
    out_stride2,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    idx0 = row // dim1
    idx1 = row % dim1

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < cols

    x_row_ptr = x_ptr + idx0 * x_stride0 + idx1 * x_stride1
    out_row_ptr = out_ptr + idx0 * out_stride0 + idx1 * out_stride1

    x = tl.load(x_row_ptr + offs * x_stride2, mask=mask, other=-float("inf")).to(tl.float32)
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    y = num / den

    tl.store(out_row_ptr + offs * out_stride2, y, mask=mask)


@torch.fx.wrap
def _softmax_unsqueeze(x):
    dim0, dim1, cols = x.shape
    out = torch.empty((dim0, dim1, cols, 1), device=x.device, dtype=x.dtype)
    block_size = 1
    while block_size < cols:
        block_size *= 2
    num_warps = 4
    if block_size >= 1024:
        num_warps = 8
    rows = dim0 * dim1
    _softmax_unsqueeze_kernel[(rows,)](
        x,
        out,
        dim1,
        cols,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return out


def replacement_func():
    return _softmax_unsqueeze