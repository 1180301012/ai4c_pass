import torch
import sys
import os

# Add pass_dir to sys.path so we can import the shared kernel
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _shared_kernel import attention_mask_fused

from torch import device

def pattern(in_0):
    tmp_1 = torch.full((9, 9), -3.4028234663852886e+38, device = device(type='cuda', index=0))
    tmp_2 = torch.arange(9, device = device(type='cuda', index=0))
    tmp_3 = tmp_2 + 1
    tmp_4 = tmp_3.view(9, 1)
    tmp_5 = tmp_2 < tmp_4
    tmp_6 = tmp_1.masked_fill_(tmp_5, 0)
    tmp_7 = tmp_1.to(torch.float32)
    tmp_8 = tmp_7[(None, None, slice(None, None, None), slice(None, None, None))]
    tmp_9 = tmp_8.expand(1, 1, 9, 9)
    tmp_10 = in_0[(slice(None, None, None), None, None, slice(None, None, None))]
    tmp_11 = tmp_10.expand(1, 1, 9, 9)
    tmp_12 = tmp_11.to(torch.float32)
    tmp_13 = torch.tensor(1.0, dtype = torch.float32)
    tmp_14 = tmp_13 - tmp_12
    tmp_15 = tmp_14.to(torch.bool)
    tmp_16 = tmp_14.masked_fill(tmp_15, -3.4028234663852886e+38)
    tmp_17 = tmp_16.to(device(type='cuda', index=0))
    tmp_18 = tmp_17.bool()
    tmp_19 = tmp_9.masked_fill(tmp_18, -3.4028234663852886e+38)
    return (tmp_19,)

def replacement_args(in_0):
    return (in_0, "size9")

def replacement_func():
    return attention_mask_fused