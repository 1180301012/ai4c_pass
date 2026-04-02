import torch
import triton
import triton.language as tl


def pattern(x):
    tmp5 = x.sum(dim=3, keepdim=True)
    tmp6 = x / tmp5
    return tmp6


def replacement_args(x):
    return (x,)


@triton.jit
def _reduce_sum_div_kernel(
    x_ptr,
    out_ptr,
    n_cols,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=0.0)
    x_f = x.to(tl.float32)
    s = tl.sum(x_f, axis=0)
    out = (x_f / s).to(x.dtype)
    tl.store(out_ptr + row * n_cols + offs, out, mask=mask)


@torch.fx.wrap
def fuse_reduce_sum_div(x):
    orig_shape = x.shape
    n_cols = orig_shape[-1]
    n_rows = x.numel() // n_cols
    xc = x.contiguous().view(n_rows, n_cols)
    out = torch.empty_like(xc)
    _reduce_sum_div_kernel[(n_rows,)](xc, out, n_cols, BLOCK=8)
    return out.view(orig_shape)


def replacement_func():
    return fuse_reduce_sum_div