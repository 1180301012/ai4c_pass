import torch
from torch import device
from pass_dir.FuseArange128CastBool import triton_cast_to_bool

def pattern(in_0):
    tmp_2 = in_0.to(device=device(type='cuda', index=0), dtype=torch.bool)
    return tmp_2

def replacement_args(in_0):
    return (in_0,)

def replacement_func():
    return triton_cast_to_bool