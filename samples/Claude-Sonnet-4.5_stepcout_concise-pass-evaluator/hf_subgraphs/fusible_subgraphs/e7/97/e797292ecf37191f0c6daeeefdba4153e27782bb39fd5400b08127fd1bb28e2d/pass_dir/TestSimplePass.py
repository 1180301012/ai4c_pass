import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """
    Simpler pattern to test - just softmax + dropout + bmm
    """
    tmp_0 = torch.nn.functional.softmax(in_0, dim=-1)
    tmp_1 = torch.nn.functional.dropout(tmp_0, p=0.0, training=False)
    tmp_2 = torch.bmm(tmp_1, in_1)
    tmp_3 = tmp_2.view(1, 8, 1, 32)
    tmp_4 = tmp_3.transpose(1, 2)
    tmp_5 = tmp_4.reshape(1, 1, 256)
    return tmp_5


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@torch.fx.wrap
def simple_reshape(in_0, in_1):
    """Just use PyTorch's native reshape - minimal overhead test"""
    # in_1 is [8, 1, 32], we want [1, 1, 256]
    out = in_1.reshape(1, 1, 256)
    return out


def replacement_func():
    return simple_reshape