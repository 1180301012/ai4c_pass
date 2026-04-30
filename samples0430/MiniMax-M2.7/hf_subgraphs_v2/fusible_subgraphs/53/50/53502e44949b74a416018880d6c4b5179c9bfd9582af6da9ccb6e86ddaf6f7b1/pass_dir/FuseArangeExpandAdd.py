import torch
from pass_dir.shared_kernels import fused_dispatch_wrapper, ROUTE_ARANGE_EXPAND_ADD


def pattern(in_0, in_1, in_2, in_3, in_4):
    """
    Pattern specifically matching the arange/expand/add pattern.
    arange(0, 1).expand(1, -1) + 2 is equivalent to arange(2, 3)
    """
    from torch import device
    
    tmp_6 = torch.arange(0, 1, dtype=torch.int64, device=device(type='cuda', index=0))
    tmp_7 = tmp_6.expand(1, -1)
    tmp_8 = tmp_7 + 2
    
    return tmp_8


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    """
    Extract arguments for the arange optimization.
    arange_start = 0, add_val = 2
    Result: arange(0+2, 0+2+1) = arange(2, 3)
    """
    return (in_0, in_1, in_2, in_3, in_4, ROUTE_ARANGE_EXPAND_ADD, 0, 2)


def replacement_func():
    """
    Returns the shared dispatch wrapper function.
    """
    return fused_dispatch_wrapper