import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Pattern matching for tensor slicing operations only
    """
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = in_2
    tmp_3 = in_3
    tmp_4 = in_5[slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, None)]
    tmp_5 = torch.nn.functional.batch_norm(in_4, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 0.001)
    return (tmp_5, tmp_4)

def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Extract arguments for the replacement kernel
    """
    return (in_0, in_1, in_2, in_3, in_4, in_5)

@torch.fx.wrap
def optimized_slice(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Optimized slice operation - reduce overhead by avoiding intermediate variables
    """
    # Direct slice operation without temporary variables
    out_5 = torch.nn.functional.batch_norm(in_4, in_0, in_1, in_3, in_2, False, 0.1, 0.001)
    out_4 = in_5  # Direct slice assignment
    
    return (out_5, out_4)

def replacement_func():
    """
    Return the optimized function
    """
    return optimized_slice