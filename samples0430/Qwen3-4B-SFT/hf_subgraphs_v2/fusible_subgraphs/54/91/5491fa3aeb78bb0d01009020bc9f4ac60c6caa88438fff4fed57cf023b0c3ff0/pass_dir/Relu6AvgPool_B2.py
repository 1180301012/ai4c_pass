"""
Pass: fused ReLU6 + global average pool for B=2.
Matches: hardtanh(in_0, 0.0, 6.0, True) -> adaptive_avg_pool2d((1,1)) -> view(2,-1) -> flatten(1)
Uses shared routing dispatcher from fused_relu6_avgpool_kernel.
"""
import torch
import triton
import triton.language as tl
from pass_dir.fused_relu6_avgpool_kernel import relu6_gavgpool_kernel, _call_relu6_gavgpool, _shared_relu6_gavgpool_dispatch


def pattern(in_0):
    tmp_0 = torch.nn.functional.hardtanh(in_0, 0.0, 6.0, True)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, (1, 1))
    tmp_2 = tmp_1.view(2, -1)
    tmp_3 = torch.flatten(tmp_2, 1)
    return (tmp_3,)


def replacement_args(in_0):
    return (in_0, "B2")


def replacement_func():
    return _shared_relu6_gavgpool_dispatch