import torch
import triton
import triton.language as tl
from pass_dir.shared_kernel import fused_add_softmax


def pattern(in_0, in_1, clamp_min):
    tmp_0 = torch.ops.aten.add.Tensor(in_1, in_0)
    tmp_2 = torch.ops.aten.maximum.default(tmp_0, clamp_min)
    return tmp_2


def replacement_args(in_0, in_1, clamp_min):
    return (in_0, in_1)


def replacement_func():
    return fused_add_softmax