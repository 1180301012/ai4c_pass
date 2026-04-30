import torch
import triton
import triton.language as tl

# This file shares the same kernel implementations and replacement_func 
# from FuseAddLayerNorm.py via import
from pass_dir.FuseAddLayerNorm import (
    _dispatch_wrapper,
    layer_norm_kernel,
    _layer_norm_kernel_wrapper,
    _add_layer_norm_fused
)


# ==============================================================================
# Pass 2: Match Hubert pattern (return norm, sum)
# ==============================================================================

def pattern(in_0, in_1, in_2, in_3):
    """
    Match pattern: add + layer_norm for Hubert models (return (norm, sum)).
    The pattern returns tmp_4 first, then tmp_2.
    """
    tmp_2 = in_2 + in_3
    tmp_4 = torch.nn.functional.layer_norm(tmp_2, (1024,), in_1, in_0, 1e-05)
    # Return norm first, then sum (matching Hubert models)
    return tmp_4, tmp_2


def replacement_args(in_0, in_1, in_2, in_3):
    """Extract arguments with route string for norm_first pattern."""
    return (in_0, in_1, in_2, in_3, "norm_first")


def replacement_func():
    """Returns the module-level replacement function (same as other pass)."""
    return _dispatch_wrapper