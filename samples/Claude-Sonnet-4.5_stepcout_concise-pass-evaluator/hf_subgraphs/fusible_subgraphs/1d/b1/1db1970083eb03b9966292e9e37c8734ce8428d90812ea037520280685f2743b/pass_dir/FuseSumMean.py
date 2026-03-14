import torch
import triton
import triton.language as tl

@torch.fx.wrap
def fused_sum_mean(in_0):
    """
    Simplified implementation - directly compute sum then mean
    """
    # Sum over dimension 1 (C), then mean over spatial dimensions (H, W) which are dims 2,3
    # This allows PyTorch to potentially fuse these operations internally
    return in_0.sum(dim=1, keepdim=False).mean(dim=(2, 3), keepdim=True)

def pattern(in_0):
    """
    Pattern to match: sum along dim 1, then mean along dims (2, 3) with keepdim
    """
    tmp_0 = in_0.sum(1)
    tmp_1 = tmp_0.mean((2, 3), keepdim=True)
    return tmp_1

def replacement_args(in_0):
    """
    Extract arguments for replacement function
    """
    return (in_0,)

def replacement_func():
    """
    Return the optimized kernel wrapper function
    """
    return fused_sum_mean