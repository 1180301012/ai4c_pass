import torch
import triton
import triton.language as tl


# Minimal test: match just torch.cat([a, b], dim=-1)
@torch.fx.wrap
def _my_cat(a, b):
    return torch.cat([a, b], dim=-1)


def pattern(a, b):
    return torch.cat([a, b], dim=-1)


def replacement_args(a, b):
    return (a, b)


def replacement_func():
    return _my_cat