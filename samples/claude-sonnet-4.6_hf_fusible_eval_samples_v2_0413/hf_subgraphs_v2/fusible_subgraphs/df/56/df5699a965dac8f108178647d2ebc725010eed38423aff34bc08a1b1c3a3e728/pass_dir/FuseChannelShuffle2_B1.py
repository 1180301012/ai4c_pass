import torch
import sys, os
_pass_dir = os.path.dirname(os.path.abspath(__file__))
if _pass_dir not in sys.path:
    sys.path.insert(0, _pass_dir)
from shared_kernels import fused_shuffle2_impl

def pattern(in_3, x):
    tmp_6  = torch.cat([in_3, x], dim=1)
    tmp_11 = tmp_6.view(1, 2, 40, 32, 24)
    tmp_12 = torch.transpose(tmp_11, 1, 2)
    tmp_13 = tmp_12.contiguous()
    tmp_14 = tmp_13.view(1, 80, 32, 24)
    return tmp_14

def replacement_args(in_3, x):
    return (in_3, x)

def replacement_func():
    return fused_shuffle2_impl