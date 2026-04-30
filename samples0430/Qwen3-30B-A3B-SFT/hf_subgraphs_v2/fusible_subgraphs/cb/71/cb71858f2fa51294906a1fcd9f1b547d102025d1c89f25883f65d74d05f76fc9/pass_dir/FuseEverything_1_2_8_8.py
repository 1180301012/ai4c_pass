"""
Combined pass: matches the ENTIRE model forward in a single pattern and
replaces it with a single fused_dispatch call, minimising dispatch + launch
overhead.

Pattern covers:
  tmp_4 = conv2d(in_2, in_1, in_0, stride=1, pad=0) → view(1,2,8,8) → sigmoid
  tmp_6 = in_3.sum(dim=3, keepdim=True) → in_3 / that_sum

Replacement: one Python call + one CUDA kernel launching both sub-computations.
"""
import torch
from pass_dir.shared_ops import fused_dispatch


def pattern(in_2, in_1, in_0, in_3):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3  = conv2d.view(1, 2, 8, 8)
    tmp_4  = tmp_3.sigmoid()
    tmp_5  = in_3.sum(dim=3, keepdim=True)
    tmp_6  = in_3 / tmp_5
    return tmp_6, tmp_4


def replacement_args(in_2, in_1, in_0, in_3):
    return (in_2, in_1, in_0, in_3, "combined")


def replacement_func():
    return fused_dispatch