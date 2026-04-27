"""
Optimization pass: replace torch.arange(0, 512, device=dev) with a Triton kernel.
"""
import torch
import triton
import triton.language as tl

from pass_dir.kernels_shared import run_arange


def pattern(dev):
    return torch.arange(0, 512, device=dev)


def replacement_args(dev):
    return (dev,)


@torch.fx.wrap
def _arange_512(dev):
    return run_arange(dev, 512)


def replacement_func():
    return _arange_512