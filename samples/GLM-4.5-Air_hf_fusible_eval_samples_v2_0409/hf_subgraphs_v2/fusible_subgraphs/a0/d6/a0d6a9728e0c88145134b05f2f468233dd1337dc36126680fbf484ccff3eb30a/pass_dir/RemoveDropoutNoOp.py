import torch
import triton
import triton.language as tl

def pattern(tmp_1):
    tmp_2 = torch.nn.functional.dropout(tmp_1, p=0.0, training=False)
    return tmp_2

def replacement_args(tmp_1):
    return (tmp_1,)

def identity_func(tmp_1):
    return tmp_1

def replacement_func():
    return identity_func