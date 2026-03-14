import torch
import triton
import triton.language as tl

def pattern(in_0):
    """Match just transpose + contiguous + reshape + contiguous"""
    tmp_5 = in_0.transpose(1, 2)
    tmp_6 = tmp_5.contiguous()
    tmp_7 = tmp_6.reshape(1, 257, -1)
    tmp_8 = tmp_7.contiguous()
    return (tmp_8,)

def replacement_args(in_0):
    return (in_0,)

@torch.fx.wrap
def optimized_transpose_reshape(x):
    """Simple wrapper that does the same thing"""
    result = x.transpose(1, 2).contiguous()
    result = result.reshape(1, 257, -1).contiguous()
    return result

def replacement_func():
    return optimized_transpose_reshape