import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_18 = torch.nn.functional.dropout(in_0, 0.1, False, False)
    return tmp_18


def replacement_args(in_0):
    return (in_0,)


@torch.fx.wrap
def identity_dropout_elim(x):
    return x


def replacement_func():
    return identity_dropout_elim