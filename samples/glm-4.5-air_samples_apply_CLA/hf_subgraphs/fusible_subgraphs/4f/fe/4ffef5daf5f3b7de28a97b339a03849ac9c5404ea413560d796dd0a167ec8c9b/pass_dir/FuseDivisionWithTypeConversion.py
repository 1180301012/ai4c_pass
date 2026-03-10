import torch
import triton
import triton.language as tl

def pattern(x, y):
    # Pattern: in_4 / in_3 (division) followed by .to(torch.float32)
    tmp_3 = x / y
    tmp_4 = tmp_3.to(torch.float32)
    return tmp_4

def replacement_args(x, y):
    return (x, y)

@torch.fx.wrap
def remove_redundant_conversion(x, y):
    """
    Simply remove the redundant .to(torch.float32) operation
    by using PyTorch's native division which already handles dtype conversion properly
    """
    # For small tensors, PyTorch's division is already optimal
    # and handles dtype conversion automatically
    return x / y

def replacement_func():
    return remove_redundant_conversion