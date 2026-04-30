"""
Pass: Replace F.linear(in_1, in_0, None) with a Triton GEMM.
Matches ALL graph variants (single-output pattern, no H-specific reshape).
"""

import torch
import triton
import triton.language as tl

from pass_dir.qkv_fused_kernel import _triton_linear_dispatch


def pattern(in_0, in_1):
    """Match only the linear operation (single output)."""
    return torch.nn.functional.linear(in_1, in_0, None)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return _triton_linear_dispatch