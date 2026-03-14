import torch
import triton
import triton.language as tl


def pattern(in_1):
    """
    Pattern for permute + reshape:
    - in_1: [1, C, H, W]
    Returns: [H*W, C]
    """
    tmp_2 = in_1.permute(0, 2, 3, 1)
    tmp_3 = tmp_2.reshape(3969, -1)
    return tmp_3


def replacement_args(in_1):
    return (in_1,)


@torch.fx.wrap
def fused_transform(in_1):
    """
    Optimized implementation: For small tensors like this,
    native PyTorch is already optimized, so just use it directly.
    """
    B, C, H, W = in_1.shape
    HW = H * W
    out = in_1.permute(0, 2, 3, 1).reshape(HW, C)
    return out


def replacement_func():
    return fused_transform