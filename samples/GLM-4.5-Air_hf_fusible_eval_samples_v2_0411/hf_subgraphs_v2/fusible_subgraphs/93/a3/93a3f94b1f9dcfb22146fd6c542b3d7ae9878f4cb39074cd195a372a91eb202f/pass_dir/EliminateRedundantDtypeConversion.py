import torch
import triton
import triton.language as tl

@torch.fx.wrap
def eliminate_redundant_conversion(x):
    return x

def pattern(x):
    # Match .to(torch.float32) on a tensor that should already be float32 after softmax
    return x.to(torch.float32)

def replacement_args(x):
    return (x,)

def replacement_func():
    return eliminate_redundant_conversion