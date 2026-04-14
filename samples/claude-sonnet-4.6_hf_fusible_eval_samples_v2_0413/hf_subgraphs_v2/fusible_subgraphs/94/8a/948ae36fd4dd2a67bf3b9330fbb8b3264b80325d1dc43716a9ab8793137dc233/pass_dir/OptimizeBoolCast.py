"""
Optimization pass: replace in_0.to(device=cuda, dtype=torch.bool) with a
Triton kernel for all graphs in the benchmark.

torch.arange is NOT matched here because FX constant-folds it (no tensor
inputs) into a get_attr/_tensor_constant0 node, making call_function-level
pattern matching crash.  The arange stays in the original graph unchanged.
"""
import torch
from torch import device
from pass_dir.arange_bool_cast_kernels import optimized_bool_cast


def pattern(in_0):
    tmp_2 = in_0.to(device=device(type='cuda', index=0), dtype=torch.bool)
    return tmp_2


def replacement_args(in_0):
    return (in_0,)


def replacement_func():
    return optimized_bool_cast