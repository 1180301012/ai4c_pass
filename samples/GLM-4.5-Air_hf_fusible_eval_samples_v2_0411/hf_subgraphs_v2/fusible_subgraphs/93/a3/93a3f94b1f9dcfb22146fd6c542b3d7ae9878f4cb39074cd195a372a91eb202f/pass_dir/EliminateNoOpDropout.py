import torch
import triton
import triton.language as tl

@torch.fx.wrap
def eliminate_no_op_dropout(x):
    return x

def pattern(x):
    # Match dropout with p=0.0 which is essentially a no-op
    return torch.nn.functional.dropout(x, p=0.0, training=False)

def replacement_args(x):
    return (x,)

def replacement_func():
    return eliminate_no_op_dropout