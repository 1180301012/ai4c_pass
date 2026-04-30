import torch
from pass_dir.shared_kernel import conv1x1_view_softmax


def pattern(bias, weight, x):
    conv2d = torch.conv2d(x, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    view = conv2d.view(32, 1, -1)
    softmax = view.softmax(dim=-1)
    return softmax


def replacement_args(bias, weight, x):
    return (bias, weight, x)


def replacement_func():
    return conv1x1_view_softmax