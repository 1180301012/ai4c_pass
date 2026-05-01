"""
Pass: fuse layer_norm with normalized_shape=(768,) and eps=1e-12
Matches float16 (yolos-base) graph.
Uses shared_dispatch routing so replacement_func() is identical across all passes.
"""
import torch
from pass_dir.triton_kernels import shared_dispatch


def pattern(x, weight, bias):
    return torch.nn.functional.layer_norm(x, (768,), weight, bias, 1e-12)


def replacement_args(x, weight, bias):
    return (x, weight, bias, "ln768")


def replacement_func():
    return shared_dispatch