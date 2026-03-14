import torch
import triton
import triton.language as tl


def pattern():
    """
    Pattern to match a simple torch.arange operation.
    """
    result = torch.arange(24)
    return result


def replacement_args():
    return ()


@torch.fx.wrap
def fused_arange_24():
    """
    Simple replacement for arange.
    """
    return torch.arange(24, device='cuda')


def replacement_func():
    return fused_arange_24