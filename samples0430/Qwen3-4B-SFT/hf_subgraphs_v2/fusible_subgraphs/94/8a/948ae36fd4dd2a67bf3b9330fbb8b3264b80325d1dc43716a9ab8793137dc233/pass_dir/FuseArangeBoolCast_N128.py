"""
Optimization pass for: in_0.to(device='cuda', dtype=torch.bool)
This pass uses the shared routing dispatch_bool_cast so all 4 variants
(FuseArangeBoolCast_N128/N256/N512/N1024) share the same replacement_func.
"""
import torch
from torch import device as _torch_device
from pass_dir.kernels import dispatch_bool_cast


def pattern(in_0):
    return in_0.to(device=_torch_device(type='cuda', index=0), dtype=torch.bool)


def replacement_args(in_0):
    return (in_0, "n128")


def replacement_func():
    return dispatch_bool_cast