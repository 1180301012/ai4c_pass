import torch
import triton
import triton.language as tl
from pass_dir.fused_bn_silu_kernel import fused_bn_silu_dispatch


def pattern(in_0, in_1, in_2, in_3, in_4):
    """Match: reshape(1, 512, 8, 8) + batch_norm(inference) + silu(inplace)"""
    tmp_4 = in_4.reshape(1, 512, 8, 8)
    tmp_5 = torch.nn.functional.batch_norm(tmp_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_6 = torch.nn.functional.silu(tmp_5, inplace=True)
    return (tmp_6,)


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    """Extract arguments for the dispatch wrapper.
    
    in_0 = running_mean
    in_1 = running_var
    in_2 = bias
    in_3 = weight
    in_4 = input tensor
    
    Route string identifies this as the 512-channel configuration.
    """
    return (in_0, in_1, in_2, in_3, in_4, "bn_silu_512_8_8")


def replacement_func():
    """Return the shared dispatch wrapper (same across all passes)."""
    return fused_bn_silu_dispatch