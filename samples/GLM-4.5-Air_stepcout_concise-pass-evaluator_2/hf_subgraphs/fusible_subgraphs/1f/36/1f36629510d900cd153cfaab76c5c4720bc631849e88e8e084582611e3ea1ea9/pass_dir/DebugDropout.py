import torch
import triton
import triton.language as tl

@torch.fx.wrap
def identity_dropout(x):
    return x

def pattern(tmp_2):
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.0, False, False)
    tmp_2 = None
    return (tmp_3,)

def replacement_args(tmp_2):
    return (tmp_2,)

def replacement_func():
    return identity_dropout