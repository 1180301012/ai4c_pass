import torch
from torch import device
from pass_dir.shared_kernels import dispatch_wrapper

# Pattern matching - matches the full computation: arange(0, 128) + cast
def pattern(in_0):
    tmp_1 = torch.arange(0, 128, device=device(type='cuda', index=0))
    tmp_2 = in_0.to(device=device(type='cuda', index=0), dtype=torch.bool)
    return (tmp_1, tmp_2)

# Argument extraction - appends route string
def replacement_args(in_0):
    return (in_0, "fuse_arange_128_cast_bool")

# Replacement function (zero-argument, returns callable)
# All passes share the SAME dispatch_wrapper function object via import
def replacement_func():
    return dispatch_wrapper