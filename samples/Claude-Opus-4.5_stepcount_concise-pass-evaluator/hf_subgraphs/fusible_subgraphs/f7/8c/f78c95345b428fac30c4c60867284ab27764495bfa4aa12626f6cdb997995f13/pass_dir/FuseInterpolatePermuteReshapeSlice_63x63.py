import torch
import triton
import triton.language as tl

# Pattern for 63x63 case - match slice operation
def pattern(in_0):
    tmp_4 = in_0[slice(3969, None, None)]
    return tmp_4

def replacement_args(in_0):
    return (in_0,)

@torch.fx.wrap
def optimized_slice_63(in_0):
    # in_0: [3972, 16] - slice from 3969 to end gives [3, 16]
    return in_0[3969:]


def replacement_func():
    return optimized_slice_63