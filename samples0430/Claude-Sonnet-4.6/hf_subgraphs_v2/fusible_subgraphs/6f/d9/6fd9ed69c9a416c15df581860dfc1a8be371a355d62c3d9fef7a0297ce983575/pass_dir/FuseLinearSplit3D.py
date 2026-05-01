"""
FuseLinearSplit3D pass
Pattern: F.linear(x, w, b) → full output  (SINGLE OUTPUT, same as FuseLinearSplitUnsqueeze)

This pass shares the same pattern and replacement function as FuseLinearSplitUnsqueeze.
After FuseLinearSplitUnsqueeze replaces all F.linear nodes, this pass is a no-op.
Both passes are listed to ensure both linear operations are covered if ordering changes.
"""
import torch
from pass_dir.linear_split_impl import dispatch_fused_linear_split  # shared obj


def pattern(x, w, b):
    return torch.nn.functional.linear(x, w, b)


def replacement_args(x, w, b):
    return (x, w, b, 1)


def replacement_func():
    return dispatch_fused_linear_split