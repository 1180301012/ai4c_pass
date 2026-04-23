import torch
import triton
import triton.language as tl

from pass_dir.shared_conv1x1_softmax import fused_conv1x1_softmax_unsqueeze as _impl


def pattern(in_0: torch.Tensor, in_1: torch.Tensor, in_2: torch.Tensor):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.view(4, 1, 192)
    tmp_4 = torch.nn.functional.softmax(tmp_3, 2, _stacklevel=5)
    tmp_5 = tmp_4.unsqueeze(-1)
    return tmp_5


def replacement_args(in_0: torch.Tensor, in_1: torch.Tensor, in_2: torch.Tensor):
    return (in_0, in_1, in_2)


@torch.fx.wrap
def fused_conv1x1_softmax_unsqueeze(in_0, in_1, in_2):
    return _impl(in_0, in_1, in_2)


def replacement_func():
    return fused_conv1x1_softmax_unsqueeze