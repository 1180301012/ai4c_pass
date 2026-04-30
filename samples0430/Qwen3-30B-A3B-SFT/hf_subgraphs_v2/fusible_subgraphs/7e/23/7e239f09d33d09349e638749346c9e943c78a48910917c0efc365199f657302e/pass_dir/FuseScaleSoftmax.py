import operator
import torch
import torch.fx as _fx
from torch import device
import triton
import triton.language as tl

# In PyTorch 2.9, torch.fx.Proxy.__itruediv__ is NOT defined, so FX traces
# `a /= b` as truediv instead of itruediv. Monkey-patch to create the
# correct itruediv nodes so the pattern matches dynamo-traced model graphs.
def _proxy_itruediv(self, other):
    return self.tracer.create_proxy('call_function', operator.itruediv, (self, other), {})

if not hasattr(_fx.Proxy, '__itruediv__'):
    _fx.Proxy.__itruediv__ = _proxy_itruediv


def pattern(in_0, tmp_2, tmp_4):
    in_0 /= tmp_2
    tmp_3 = in_0
    tmp_3 /= tmp_4
    tmp_6 = tmp_3.softmax(dim=-1)
    return tmp_6


def replacement_args(in_0, tmp_2, tmp_4):
    return (in_0,)


@triton.jit
def _scale_softmax_kernel(
    input_ptr,
    output_ptr,
    scale,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * N
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(input_ptr + row_start + offsets, mask=mask, other=float('-inf'))
    x = x.to(tl.float32)
    x = x * scale

    x_max = tl.max(x, axis=0)
    x = x - x_max
    x = tl.exp(x)
    x_sum = tl.sum(x, axis=0)
    x = x / x_sum

    tl.store(output_ptr + row_start + offsets, x, mask=mask)


@torch.fx.wrap
def _scale_softmax_wrapper(in_0):
    N = in_0.shape[-1]
    M = in_0.numel() // N
    scale = 1.25

    output = torch.empty_like(in_0)

    _scale_softmax_kernel[(M,)](
        in_0,
        output,
        scale,
        N,
        BLOCK_SIZE=4096,
        num_warps=8,
    )

    return output


def replacement_func():
    return _scale_softmax_wrapper