"""
FuseLinearSplitUnsqueeze pass
Pattern: F.linear(x, w, b) → full output [M, N]  (SINGLE OUTPUT)

Using a single-output pattern is required because the backend's
_replace_pattern asserts len(match.returning_nodes)==len(copied_returning_nodes).
The replacement's with_dispatch_wrapper_run always creates ONE FX node, so the
pattern must also return exactly ONE node.

Downstream slice / view / unsqueeze ops remain in the graph as zero-copy
metadata ops and are not part of this pattern.
"""
import torch
from pass_dir.linear_split_impl import dispatch_fused_linear_split  # shared obj


def pattern(x, w, b):
    return torch.nn.functional.linear(x, w, b)


def replacement_args(x, w, b):
    return (x, w, b, 0)


def replacement_func():
    return dispatch_fused_linear_split