import torch
from torch import device
from pass_dir.shared_kernels import fused_arange_cast_dispatch

def pattern(in_0):
    tmp_1 = torch.arange(0, 512, device = device(type='cuda', index=0))
    tmp_2 = in_0.to(device = device(type='cuda', index=0), dtype = torch.bool)
    return (tmp_1, tmp_2)

def replacement_args(in_0):
    return (in_0, "arange_512")

def replacement_func():
    return fused_arange_cast_dispatch