import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """
    Simple test pattern - just add + softmax
    """
    tmp_0 = in_0 + in_1
    tmp_1 = torch.nn.functional.softmax(tmp_0, dim=-1)
    return tmp_1


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    # Just return None for now - we want to see if pattern matches
    def simple_add_softmax(in_0, in_1):
        tmp_0 = in_0 + in_1
        return torch.nn.functional.softmax(tmp_0, dim=-1)
    return simple_add_softmax