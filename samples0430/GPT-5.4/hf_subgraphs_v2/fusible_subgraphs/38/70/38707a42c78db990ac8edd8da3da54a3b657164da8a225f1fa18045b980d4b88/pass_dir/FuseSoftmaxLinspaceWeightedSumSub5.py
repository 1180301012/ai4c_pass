import inspect
import operator

import torch
import triton
import triton.language as tl
from torch import device


def pattern():
    graph = torch.fx.Graph()
    in_0 = graph.placeholder("in_0")
    tmp_0 = graph.call_function(torch.nn.functional.softmax, args=(in_0,), kwargs={"dim": 1})
    tmp_1 = graph.call_function(torch.linspace, args=(0, 4), kwargs={"steps": 5, "device": device(type="cuda", index=0)})
    tmp_2 = graph.call_function(operator.mul, args=(tmp_0, tmp_1), kwargs={})
    tmp_3 = graph.call_method("sum", args=(tmp_2,), kwargs={"dim": 1})
    tmp_4 = graph.call_function(operator.sub, args=(5, tmp_3), kwargs={})
    graph.output(tmp_4)
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    gm.__signature__ = inspect.Signature(
        parameters=[inspect.Parameter("in_0", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    )
    return gm


pattern = pattern()


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def _softmax_weighted_sum_sub5_kernel(
    x_ptr,
    out_ptr,
    stride_x0,
    stride_x1,
    stride_out0,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < 5
    x = tl.load(x_ptr + pid * stride_x0 + offs * stride_x1, mask=mask, other=-float("inf"))
    x = x.to(tl.float32)
    row_max = tl.max(x, axis=0)
    exp_x = tl.exp(x - row_max)
    denom = tl.sum(exp_x, axis=0)
    numer = tl.sum(exp_x * offs.to(tl.float32), axis=0)
    result = 5.0 - numer / denom
    tl.store(out_ptr + pid * stride_out0, result)


@torch.fx.wrap
def fused_softmax_linspace_weighted_sum_sub5(in_0):
    n_rows = in_0.shape[0]
    out = torch.empty((n_rows,), device=in_0.device, dtype=torch.float32)
    _softmax_weighted_sum_sub5_kernel[(n_rows,)](
        in_0,
        out,
        in_0.stride(0),
        in_0.stride(1),
        out.stride(0),
        BLOCK_SIZE=8,
        num_warps=1,
        num_stages=1,
    )
    return out


def replacement_func():
    return fused_softmax_linspace_weighted_sum_sub5