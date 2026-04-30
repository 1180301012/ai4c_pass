import torch
from torch import device
from pass_dir.shared_kernels import dispatch_wrapper

# Pattern matching - matches the dtype cast operation
def pattern(in_0):
    result = in_0.to(device=device(type='cuda', index=0), dtype=torch.bool)
    return result

# Argument extraction - appends route string
def replacement_args(in_0):
    return (in_0, "cast_int64_to_bool")

# Replacement function (zero-argument, returns callable)
# All passes share the SAME dispatch_wrapper function object via import
def replacement_func():
    return dispatch_wrapper