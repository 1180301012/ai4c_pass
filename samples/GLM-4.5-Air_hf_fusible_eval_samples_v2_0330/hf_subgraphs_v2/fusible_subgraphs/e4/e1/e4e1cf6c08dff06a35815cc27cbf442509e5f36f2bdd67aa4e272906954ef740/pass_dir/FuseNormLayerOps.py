import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    """Pattern matches the complete computation that returns both tmp_2 and tmp_13."""
    # tmp_2 is used in return and tmp_13 computation
    tmp_2 = in_0 * in_2
    # tmp_13 computation using in_1
    tmp_10 = in_1.float()
    tmp_11 = 1.0 + tmp_10
    tmp_13 = tmp_11.type_as(tmp_2)
    return tmp_2, tmp_13

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@torch.fx.wrap
def optimized_operations(in_0, in_1, in_2):
    """Optimized wrapper that computes both result values efficiently."""
    # Compute tmp_2 = in_0 * in_2
    tmp_2 = in_0 * in_2
    
    # Compute tmp_13 = (1.0 + in_1.float()).type_as(tmp_2)
    tmp_10 = in_1.float()
    tmp_11 = 1.0 + tmp_10
    tmp_13 = tmp_11.type_as(tmp_2)
    
    return tmp_2, tmp_13

def replacement_func():
    return optimized_operations