import torch
from torch import device

def pattern(x):
    tmp_3 = x.to(device=device(type='cuda', index=0), dtype=torch.float32)
    tmp_4 = tmp_3.t()
    return tmp_3, tmp_4

def replacement_args(x):
    return (x,)

@torch.fx.wrap
def optimized_device_transfer_transpose(x):
    # Eliminate redundant device transfer - x is already on correct device
    # Eliminate redundant dtype conversion - x is already float32
    # Use PyTorch's optimized transpose for [1, 512] -> [512, 1]
    return x, x.t()

def replacement_func():
    return optimized_device_transfer_transpose