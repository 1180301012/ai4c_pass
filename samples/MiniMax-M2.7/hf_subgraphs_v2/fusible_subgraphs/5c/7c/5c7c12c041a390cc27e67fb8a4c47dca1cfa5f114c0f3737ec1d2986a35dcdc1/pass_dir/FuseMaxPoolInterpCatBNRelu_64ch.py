"""
Pass for 64-channel ERFNet pattern (start7_end12_1).
Fuses: max_pool2d + interpolate + cat + batch_norm + relu
"""

import torch
from pass_dir.shared_kernels import fused_dispatch


def pattern(in_4, in_5, mean, var, weight, bias):
    """
    Pattern for start7_end12_1 (64 channels).
    in_4: [B, 48, 128, 128], in_5: [B, 16, 256, 256]
    
    max_pool2d(2,2) reduces 256x256 to 128x128
    interpolate(128,128) upsamples back to 128x128
    """
    tmp_4 = torch.nn.functional.max_pool2d(in_5, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
    tmp_5 = torch.nn.functional.interpolate(tmp_4, (128, 128), None, 'bilinear', False)
    tmp_6 = torch.cat([in_4, tmp_5], 1)
    tmp_7 = torch.nn.functional.batch_norm(tmp_6, mean, var, weight, bias, False, 0.1, 0.001)
    tmp_8 = torch.nn.functional.relu(tmp_7, inplace=False)
    return tmp_8


def replacement_args(in_4, in_5, mean, var, weight, bias):
    """Extract arguments and route for 64-channel pattern."""
    return (in_4, in_5, mean, var, weight, bias, "64ch")


def replacement_func():
    """Returns the shared dispatcher."""
    return fused_dispatch