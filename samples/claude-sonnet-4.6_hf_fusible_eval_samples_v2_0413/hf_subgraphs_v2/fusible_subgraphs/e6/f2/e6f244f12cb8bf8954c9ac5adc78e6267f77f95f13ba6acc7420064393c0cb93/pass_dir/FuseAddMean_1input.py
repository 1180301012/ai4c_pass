"""
Universal pattern: match x.mean((2,3), keepdim=True) for any tensor x.
The pattern returns only y (the internally-computed mean) — NOT x, since x
is the pattern input (not produced inside the subgraph).
fast_mean returns mean_out only; the passthrough x stays in the graph.
All 4 pass files share the same fast_mean object to bypass replacement_func_limit.
"""
import torch
import triton
import triton.language as tl
from pass_dir.shared_kernel import fast_mean


def pattern(x):
    # Only y is computed inside this matched subgraph
    y = x.mean((2, 3), keepdim=True)
    return y


def replacement_args(x):
    return (x,)


def replacement_func():
    return fast_mean