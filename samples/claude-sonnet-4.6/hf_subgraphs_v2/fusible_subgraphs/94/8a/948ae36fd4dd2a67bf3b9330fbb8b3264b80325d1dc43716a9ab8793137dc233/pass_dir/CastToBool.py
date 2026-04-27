"""
Optimization pass: replace in_0.to(device=dev, dtype=torch.bool) with a Triton kernel.

The `dev` placeholder captures the device constant (stored as a get_attr node by Dynamo).
Single-return pattern avoids tuple-output complexity.
"""
import torch
import triton
import triton.language as tl

from pass_dir.kernels_shared import run_cast_bool


def pattern(in_0, dev):
    return in_0.to(device=dev, dtype=torch.bool)


def replacement_args(in_0, dev):
    return (in_0,)


@torch.fx.wrap
def _cast_to_bool(in_0):
    return run_cast_bool(in_0)


def replacement_func():
    return _cast_to_bool