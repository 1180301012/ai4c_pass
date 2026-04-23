import torch
import triton
import triton.language as tl
from graph_net_bench.torch.posion_dispatch_tensor import unwrap_tensor


_SCALE = 0.1767766952966369


# Pattern matching function
# Mirrors model.py exactly: mul by scalar -> softmax(dim=-1) -> transpose(-2, -1)
def pattern(in_0):
    tmp_0 = in_0 * _SCALE
    tmp_1 = tmp_0.softmax(dim=-1)
    tmp_2 = tmp_1.transpose(-2, -1)
    return tmp_2


# Argument extraction function
def replacement_args(in_0):
    return (in_0,)


@triton.jit
def _scaled_softmax_kernel(
    x_ptr,
    out_ptr,
    stride_0,
    stride_1,
    stride_2,
    stride_3,
    dim1,
    dim2,
    dim3,
    scale,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)

    rows_per_d0 = dim1 * dim2
    i0 = row_id // rows_per_d0
    rem = row_id % rows_per_d0
    i1 = rem // dim2
    i2 = rem % dim2

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < dim3

    x_row_ptr = x_ptr + i0 * stride_0 + i1 * stride_1 + i2 * stride_2 + offsets * stride_3
    x = tl.load(x_row_ptr, mask=mask, other=-float("inf"))
    x = x.to(tl.float32) * scale

    x_max = tl.max(x, axis=0)
    x = x - x_max
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    y = num / den

    out_row_ptr = out_ptr + row_id * dim3 + offsets
    tl.store(out_row_ptr, y, mask=mask)


@torch.fx.wrap
def fused_scale_softmax_transpose(in_0):
    x = unwrap_tensor(in_0)
    y = (x * _SCALE).softmax(dim=-1)
    return y.transpose(-2, -1)


# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_scale_softmax_transpose