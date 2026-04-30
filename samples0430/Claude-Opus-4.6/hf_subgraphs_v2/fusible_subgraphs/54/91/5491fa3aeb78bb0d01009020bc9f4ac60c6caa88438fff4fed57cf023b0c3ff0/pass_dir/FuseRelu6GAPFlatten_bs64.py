import torch
import triton
import triton.language as tl
from pass_dir._kernel import fused_relu6_gap_flatten


def pattern(in_0):
    tmp_0 = torch.nn.functional.hardtanh(in_0, 0.0, 6.0, True)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, (1, 1))
    tmp_2 = tmp_1.view(64, -1)
    tmp_3 = torch.flatten(tmp_2, 1)
    return tmp_3


def replacement_args(in_0):
    return (in_0,)


def replacement_func():
    return fused_relu6_gap_flatten