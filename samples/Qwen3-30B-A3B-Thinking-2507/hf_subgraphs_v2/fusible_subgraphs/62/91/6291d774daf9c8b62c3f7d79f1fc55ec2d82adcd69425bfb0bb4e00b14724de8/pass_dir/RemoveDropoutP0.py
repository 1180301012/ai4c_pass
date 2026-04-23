import torch
import triton
import triton.language as tl
def pattern(x):
    return torch.nn.functional.dropout(x, p=0.0, training=False)
def replacement_args(x):
    return (x, )
@torch.fx.wrap
def no_op(x):
    return x
def replacement_func():
    return no_op