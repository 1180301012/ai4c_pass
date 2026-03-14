import torch
import triton
import triton.language as tl


def pattern(in_0):
    """
    Match the pattern: flatten(2) followed by permute(0, 2, 1)
    Input shape: [B, C, H, W]
    After flatten(2): [B, C, H*W]
    After permute(0, 2, 1): [B, H*W, C]
    """
    tmp_0 = in_0.flatten(2)
    tmp_1 = tmp_0.permute(0, 2, 1)
    return tmp_1


def replacement_args(in_0):
    return (in_0,)


def flatten_permute_optimized(x):
    """
    Optimized version using permute + reshape.
    Fuses two operations into one with better memory access pattern.
    """
    B, C, H, W = x.shape
    # x.permute(0, 2, 3, 1) gives [B, H, W, C]
    # reshape to [B, H*W, C] gives the same result as flatten + permute
    return x.permute(0, 2, 3, 1).reshape(B, H * W, C)


@torch.fx.wrap
def flatten_permute_wrapper(x):
    """Wrapper function using PyTorch's optimized ops"""
    return flatten_permute_optimized(x)


def replacement_func():
    return flatten_permute_wrapper