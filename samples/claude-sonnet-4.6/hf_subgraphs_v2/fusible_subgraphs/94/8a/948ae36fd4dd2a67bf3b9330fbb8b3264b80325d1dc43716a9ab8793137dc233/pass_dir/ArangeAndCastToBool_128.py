"""
Optimization pass: replace torch.arange(0, 128, device=dev) with a Triton kernel.
Single-return pattern (only arange, cast handled separately by CastToBool pass).
"""
import torch
import triton
import triton.language as tl

from pass_dir.kernels_shared import run_arange


def pattern(dev):
    return torch.arange(0, 128, device=dev)


def replacement_args(dev):
    return (dev,)


@torch.fx.wrap
def _arange_128(dev):
    return run_arange(dev, 128)


def replacement_func():
    return _arange_128