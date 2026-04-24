"""
Pass: Replace layer_norm with normalized_shape=(768,) with a Triton kernel.
Matches float16 graphs.
"""
import torch
from pass_dir.shared_kernels import layer_norm_triton


def pattern(in_0, in_1, in_4):
    return torch.nn.functional.layer_norm(in_4, (768,), in_1, in_0, 1e-12)


def replacement_args(in_0, in_1, in_4):
    return (in_0, in_1, in_4)


def replacement_func():
    return layer_norm_triton