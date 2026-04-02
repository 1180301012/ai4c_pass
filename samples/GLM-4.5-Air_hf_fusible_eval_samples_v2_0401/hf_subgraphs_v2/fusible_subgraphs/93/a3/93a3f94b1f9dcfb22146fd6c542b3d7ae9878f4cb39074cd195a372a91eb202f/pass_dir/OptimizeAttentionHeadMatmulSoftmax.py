import torch
import triton
import triton.language as tl

def pattern(tmp_2):
    """
    Pattern matching the redundant type conversion:
    tmp_2 (softmax output) -> to(float32)
    """
    tmp_3 = tmp_2.to(torch.float32)
    return tmp_3

def replacement_args(tmp_2):
    return (tmp_2,)



@torch.fx.wrap
def remove_redundant_conversion(tmp_2):
    """
    Remove redundant type conversion - tmp_2 is already float32
    from the softmax operation, so converting to float32 again is unnecessary
    """
    return tmp_2

def replacement_func():
    return remove_redundant_conversion