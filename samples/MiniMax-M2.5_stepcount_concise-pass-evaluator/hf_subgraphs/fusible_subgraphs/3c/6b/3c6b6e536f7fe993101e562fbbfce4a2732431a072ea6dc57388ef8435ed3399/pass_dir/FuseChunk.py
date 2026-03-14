import torch
import triton
import triton.language as tl


def pattern(in_3):
    """
    Pattern: chunk in_3 along last dimension
    Matches: tmp_8 = in_3.chunk(2, dim=-1); tmp_9 = tmp_8[0]; tmp_10 = tmp_8[1]
    Returns: tmp_9, tmp_10
    """
    tmp_8 = in_3.chunk(2, dim=-1)
    tmp_9 = tmp_8[0]
    tmp_10 = tmp_8[1]
    return tmp_9, tmp_10


def replacement_args(in_3):
    return (in_3,)


@torch.fx.wrap
def fused_chunk(in_3):
    """Fused implementation for chunk."""
    tmp_9, tmp_10 = in_3.chunk(2, dim=-1)
    return tmp_9, tmp_10


def replacement_func():
    return fused_chunk