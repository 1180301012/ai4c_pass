import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    tmp_1 = in_1.cumsum(-1)
    tmp_2 = tmp_1 - 1
    tmp_3 = in_0.__eq__(0)
    # tmp_4 is computed but never used - this is the optimization opportunity
    return tmp_2, tmp_3

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@torch.fx.wrap
def eliminate_redundant_masked_fill(in_0, in_1):
    tmp_1 = in_1.cumsum(-1)
    tmp_2 = tmp_1 - 1
    tmp_3 = in_0.__eq__(0)
    # tmp_4 is computed but not used, so we skip it
    return tmp_2, tmp_3

def replacement_func():
    return eliminate_redundant_masked_fill