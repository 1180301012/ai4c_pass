import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
    ],
    key=['IS_BF16'],
)
@triton.jit
def _softmax_lastdim64_kernel(
    x_ptr,
    out_ptr,
    row_stride_x,
    row_stride_out,
    IS_BF16: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < 64

    x = tl.load(x_ptr + pid * row_stride_x + offs, mask=mask, other=-float('inf'))
    x = x.to(tl.float32)
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    out = num / den
    out_cast = out.to(tl.bfloat16) if IS_BF16 else out.to(tl.float16)
    tl.store(out_ptr + pid * row_stride_out + offs, out_cast, mask=mask)


@torch.fx.wrap
def triton_softmax_dropout_lastdim64(x):
    x = x.contiguous()
    x2 = x.view(-1, 64)
    out = torch.empty_like(x)
    out2 = out.view(-1, 64)
    n_rows = x2.shape[0]
    _softmax_lastdim64_kernel[(n_rows,)](
        x2,
        out2,
        x2.stride(0),
        out2.stride(0),
        IS_BF16=(x.dtype == torch.bfloat16),
        BLOCK=64,
    )
    return (out,)


def pattern(x):
    tmp_21 = torch.nn.functional.softmax(x, dim = -1)
    tmp_22 = torch.nn.functional.dropout(tmp_21, 0.0, False, False)
    return (tmp_22,)


def replacement_args(x):
    return (x,)


def replacement_func():
    return triton_softmax_dropout_lastdim64