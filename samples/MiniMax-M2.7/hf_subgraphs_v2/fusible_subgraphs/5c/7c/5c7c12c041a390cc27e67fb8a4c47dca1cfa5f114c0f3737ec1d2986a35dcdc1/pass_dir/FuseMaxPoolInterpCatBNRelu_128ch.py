"""
Pass for 128-channel ERFNet pattern (start73_end78_8).
Fuses: max_pool2d + interpolate + cat + batch_norm + relu
"""

import torch
from pass_dir.shared_kernels import fused_dispatch


def pattern(in_4, in_5, mean, var, weight, bias):
    """
    Pattern for start73_end78_8 (128 channels).
    in_4: [B, 64, 64, 64], in_5: [B, 64, 128, 128]
    
    max_pool2d(2,2) reduces 128x128 to 64x64
    interpolate(64,64) upsamples back to 64x64
    """
    tmp_4 = torch.nn.functional.max_pool2d(in_5, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
    tmp_5 = torch.nn.functional.interpolate(tmp_4, (64, 64), None, 'bilinear', False)
    tmp_6 = torch.cat([in_4, tmp_5], 1)
    tmp_7 = torch.nn.functional.batch_norm(tmp_6, mean, var, weight, bias, False, 0.1, 0.001)
    tmp_8 = torch.nn.functional.relu(tmp_7, inplace=False)
    return tmp_8


def replacement_args(in_4, in_5, mean, var, weight, bias):
    """Extract arguments and route for 128-channel pattern."""
    return (in_4, in_5, mean, var, weight, bias, "128ch")


def replacement_func():
    """Returns the shared dispatcher."""
    return fused_dispatch