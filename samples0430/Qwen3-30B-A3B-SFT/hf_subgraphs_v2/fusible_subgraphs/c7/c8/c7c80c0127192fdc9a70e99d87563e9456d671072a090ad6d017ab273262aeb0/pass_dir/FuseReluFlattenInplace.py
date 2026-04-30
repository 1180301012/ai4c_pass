import torch
import triton
import triton.language as tl
from pass_dir.shared_relu_dispatch import relu_fuse_dispatch


# Diagnostic: relu with inplace=True as method-style
def pattern(in_0):
    return torch.nn.functional.relu(in_0, inplace=True)


def replacement_args(in_0):
    return (in_0, "route_d")


def replacement_func():
    return relu_fuse_dispatch