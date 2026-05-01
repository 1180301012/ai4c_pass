"""
Replaces x.mean((2, 3), keepdim=True) with a Triton kernel that:
  - Accumulates in float32 for numerical stability
  - Returns same dtype and shape (B, C, 1, 1) as PyTorch's mean

This single pattern matches ALL target graphs:
  - mobileone_s4.apple_in1k_start356_end359_12 (0+in_0+0 → mean)
  - mobileone_s4.apple_in1k_start228_end231_3  (0+in_1+in_0 → mean)
  - repvgg_d2se.rvgg_in1k_start113_end116_8    (in_1+in_2+in_0 → mean)
  - repvgg_d2se.rvgg_in1k_start181_end184_14   (in_0+in_1+in_2 → mean)
"""

import torch
from pass_dir.shared_kernels import triton_spatial_mean


def pattern(x):
    return x.mean((2, 3), keepdim=True)


def replacement_args(x):
    return (x,)


def replacement_func():
    return triton_spatial_mean