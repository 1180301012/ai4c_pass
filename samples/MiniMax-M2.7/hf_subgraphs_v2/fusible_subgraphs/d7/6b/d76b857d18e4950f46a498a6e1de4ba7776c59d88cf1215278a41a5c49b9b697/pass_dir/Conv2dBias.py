"""
Pass for optimizing 1x1 Conv2d with bias using a Triton kernel.

Pattern: tmp_2 = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
- in_2: input tensor (N, K, H, W)
- in_1: weight tensor (R, K, 1, 1)
- in_0: bias tensor (R,)
- Output: (N, R, H, W)
"""

import torch
from pass_dir.SharedKernelDispatch import dispatch_kernel


def pattern(in_0, in_1, in_2):
    """
    Match the conv2d with bias pattern from the model.
    
    Pattern (from model.py):
        tmp_0 = in_0  # bias
        tmp_1 = in_1  # weight
        tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    
    Note: Arguments order in torch.conv2d is:
        torch.conv2d(input, weight, bias, stride, padding, dilation, groups)
    """
    tmp_2 = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    return tmp_2


def replacement_args(in_0, in_1, in_2):
    """
    Extract arguments for the replacement function.
    Route string "conv2d" is appended to identify this operation.
    """
    return (in_2, in_1, in_0, 1, 1, "conv2d")


def replacement_func():
    """
    Returns the shared dispatch wrapper function.
    The dispatch will route to the conv2d kernel based on route string.
    """
    return dispatch_kernel