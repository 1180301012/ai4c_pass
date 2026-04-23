"""
Pass for 16-channel ERFNet pattern (start1_end6_0).
Fuses: max_pool2d + interpolate + cat + batch_norm + relu
"""

import torch
from pass_dir.shared_kernels import fused_dispatch


def pattern(in_4, in_5, mean, var, weight, bias):
    """
    Pattern for start1_end6_0 (16 channels).
    in_4: [B, 13, 256, 256], in_5: [B, 3, 512, 512]
    
    max_pool2d(2,2) reduces 512x512 to 256x256
    interpolate(256,256) keeps size (identity for bilinear to same size)
    """
    tmp_5 = torch.nn.functional.max_pool2d(in_5, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
    tmp_6 = torch.nn.functional.interpolate(tmp_5, (256, 256), None, 'bilinear', False)
    tmp_7 = torch.cat([in_4, tmp_6], 1)
    tmp_8 = torch.nn.functional.batch_norm(tmp_7, mean, var, weight, bias, False, 0.1, 0.001)
    tmp_9 = torch.nn.functional.relu(tmp_8, inplace=False)
    return tmp_9


def replacement_args(in_4, in_5, mean, var, weight, bias):
    """Extract arguments and route for 16-channel pattern."""
    return (in_4, in_5, mean, var, weight, bias, "16ch")


def replacement_func():
    """Returns the shared dispatcher."""
    return fused_dispatch