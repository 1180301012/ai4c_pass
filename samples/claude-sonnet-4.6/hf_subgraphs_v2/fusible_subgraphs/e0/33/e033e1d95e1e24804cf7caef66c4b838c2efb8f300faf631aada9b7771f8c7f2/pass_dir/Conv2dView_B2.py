import torch
from pass_dir.conv1x1_shared import dispatch_wrapper


# Pattern: 1x1 conv2d  → view(2, 256, -1)   (float16 / batch=2)
def pattern(in_0, in_1, in_3):
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3  = conv2d.view(2, 256, -1)
    return tmp_3


def replacement_args(in_0, in_1, in_3):
    # route "conv2d_view": a=bias, b=weight, c=x
    return (in_0, in_1, in_3, "conv2d_view")


def replacement_func():
    return dispatch_wrapper