import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """Minimal test pattern"""
    tmp_0 = torch.nn.functional.softmax(in_1, dim=1)
    return tmp_0


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@torch.fx.wrap
def test_replacement(in_0, in_1):
    # Just return the original softmax for now
    return torch.nn.functional.softmax(in_1, dim=1)


def replacement_func():
    return test_replacement