import torch
import triton
import triton.language as tl

from pass_dir.shared_conv1x1_softmax import fused_conv1x1_softmax_unsqueeze


def pattern(in_0: torch.Tensor, in_1: torch.Tensor, in_2: torch.Tensor):
    conv2d = torch.ops.aten.conv2d.default(in_2, in_1, in_0, [1, 1], [0, 0], [1, 1], 1)
    tmp_3 = torch.ops.aten.view.default(conv2d, [4, 1, 192])
    tmp_4 = torch.ops.aten._softmax.default(tmp_3, 2, False)
    tmp_5 = torch.ops.aten.unsqueeze.default(tmp_4, -1)
    return tmp_5


def replacement_args(in_0: torch.Tensor, in_1: torch.Tensor, in_2: torch.Tensor):
    return (in_0, in_1, in_2)


def replacement_func():
    return fused_conv1x1_softmax_unsqueeze