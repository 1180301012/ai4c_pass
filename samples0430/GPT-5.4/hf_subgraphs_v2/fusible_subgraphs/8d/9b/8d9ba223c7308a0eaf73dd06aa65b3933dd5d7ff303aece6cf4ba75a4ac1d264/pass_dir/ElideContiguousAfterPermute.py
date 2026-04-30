import torch
import triton
import triton.language as tl

from pass_dir.shared_simple_routes import shared_simple_route


def pattern(x):
    y = x.permute(2, 0, 1)
    z = y.contiguous()
    return z


def replacement_args(x):
    y = x.permute(2, 0, 1)
    return (y, 'contiguous_identity')


def replacement_func():
    return shared_simple_route