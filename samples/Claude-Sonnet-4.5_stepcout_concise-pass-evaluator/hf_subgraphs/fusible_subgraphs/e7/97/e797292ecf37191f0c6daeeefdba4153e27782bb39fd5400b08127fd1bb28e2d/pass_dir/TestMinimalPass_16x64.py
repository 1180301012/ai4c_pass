import torch
import triton
import triton.language as tl


def pattern(x):
    """
    Test minimal pattern for graph 2 - just view -> transpose -> reshape
    For [16, 1, 64] -> [1, 1, 1024]
    """
    tmp_3 = x.view(1, 16, 1, 64)
    tmp_4 = tmp_3.transpose(1, 2)
    tmp_5 = tmp_4.reshape(1, 1, 1024)
    return tmp_5


def replacement_args(x):
    return (x,)


@torch.fx.wrap
def simple_reshape_op_16x64(x):
    """Just use PyTorch's native reshape"""
    out = x.reshape(1, 1, 1024)
    return out


def replacement_func():
    return simple_reshape_op_16x64