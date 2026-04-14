import torch
import sys, os
_pass_dir = os.path.dirname(os.path.abspath(__file__))
if _pass_dir not in sys.path:
    sys.path.insert(0, _pass_dir)
from shared_kernels import fused_shuffle1_impl

def pattern(in_2, in_4):
    tmp_5 = torch.cat([in_2, in_4], dim=1)
    tmp_7 = tmp_5.view(8, 2, 20, 64, 48)
    tmp_8 = torch.transpose(tmp_7, 1, 2)
    tmp_9 = tmp_8.contiguous()
    tmp_10 = tmp_9.view(8, 40, 64, 48)
    return tmp_10

def replacement_args(in_2, in_4):
    return (in_2, in_4)

def replacement_func():
    return fused_shuffle1_impl