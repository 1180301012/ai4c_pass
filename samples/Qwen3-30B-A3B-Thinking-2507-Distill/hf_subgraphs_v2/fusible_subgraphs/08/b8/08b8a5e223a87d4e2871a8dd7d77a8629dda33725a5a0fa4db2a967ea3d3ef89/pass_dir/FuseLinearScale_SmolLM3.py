"""
Pass: FuseLinearScale_SmolLM3  (routing: "fused")
Fused: linear(in_1, in_0) * in_2 → Triton GEMM + elementwise scale kernel.

Pattern (bfloat16 model):
    linear = torch.nn.functional.linear(in_1, in_0, None)
    tmp_2  = in_2 * linear
    return tmp_2

Pattern (float32 model):
    tmp_0  = in_0
    tmp_1  = torch.nn.functional.linear(in_1, tmp_0, None)
    tmp_2  = in_2 * tmp_1
    return tmp_2

Args: in_0=weight[N,K], in_1=input[...,K], in_2=scale[...,N]
dispatch(weight, scale, input, route) = (input @ weight.T) * scale

The linear-only pass (FuseLinearScale_gemma) runs AFTER this pass.
"""
import torch
import triton
import triton.language as tl
from pass_dir.dispatch import dispatch


def pattern(in_0, in_1, in_2):
    linear = torch.nn.functional.linear(in_1, in_0, None)
    tmp_2  = in_2 * linear
    return tmp_2


def replacement_args(in_0, in_1, in_2):
    # dispatch(weight, scale, route) – for "fused" route:
    # a=weight, b=scale; dispatch uses b as dummy input when route=="fused"
    return (in_0, in_2, "fused")


def replacement_func():
    return dispatch