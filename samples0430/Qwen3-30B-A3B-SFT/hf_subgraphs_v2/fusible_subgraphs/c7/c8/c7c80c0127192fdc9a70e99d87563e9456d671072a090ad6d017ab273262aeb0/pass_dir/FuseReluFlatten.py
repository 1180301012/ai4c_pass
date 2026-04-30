import torch
import triton
import triton.language as tl
from pass_dir.shared_relu_dispatch import relu_fuse_dispatch


# Matching .flatten(1,-1) — absorbed by placeholder so works for relu+flatten subgraph
def pattern(in_0):
    return in_0.flatten(1, -1)


def replacement_args(in_0):
    return (in_0, "route_d")


def replacement_func():
    return relu_fuse_dispatch