import torch
from pass_dir.fused_kernel import fused_conv2d_view_softmax_unsqueeze


def pattern(in_0: torch.Tensor, in_1: torch.Tensor, in_2: torch.Tensor):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.view(4, 1, 192)
    tmp_4 = torch.nn.functional.softmax(tmp_3, 2, _stacklevel=5)
    tmp_5 = tmp_4.unsqueeze(-1)
    return (tmp_5,)


def replacement_args(in_0: torch.Tensor, in_1: torch.Tensor, in_2: torch.Tensor):
    return (in_2, in_1, in_0)


def replacement_func():
    return fused_conv2d_view_softmax_unsqueeze