import torch
from torch import device
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """
    Pattern matches: in_0 / 8.0 + in_1.to(device(...))
    Removes redundant device movement for already-cuda tensors
    """
    tmp_0 = in_0 / 8.0
    tmp_1 = in_1.to(device(type='cuda', index=0))  # REDUNDANT if in_1 already on CUDA
    tmp_2 = tmp_0 + tmp_1
    return (tmp_2,)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@torch.fx.wrap
def remove_device_move_wrapper(x, y):
    """Wrapper that removes redundant device movement"""
    # Simply remove the redundant .to(device(...)) call
    # since both inputs are already on CUDA as per weight_meta
    tmp_0 = x / 8.0
    tmp_2 = tmp_0 + y  # Use y directly, no redundant device movement
    return tmp_2

def replacement_func():
    return remove_device_move_wrapper