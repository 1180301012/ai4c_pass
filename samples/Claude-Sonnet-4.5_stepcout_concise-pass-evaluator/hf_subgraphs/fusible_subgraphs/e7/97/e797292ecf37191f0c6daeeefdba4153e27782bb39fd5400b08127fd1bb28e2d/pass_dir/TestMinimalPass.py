import torch
import triton
import triton.language as tl


def pattern(x):
    """
    Test minimal pattern - just view -> transpose -> reshape
    """
    tmp_3 = x.view(1, 8, 1, 32)
    tmp_4 = tmp_3.transpose(1, 2)
    tmp_5 = tmp_4.reshape(1, 1, 256)
    return tmp_5


def replacement_args(x):
    return (x,)


@torch.fx.wrap
def simple_reshape_op(x):
    """Just use PyTorch's native reshape"""
    out = x.reshape(1, 1, 256)
    return out


def replacement_func():
    return simple_reshape_op