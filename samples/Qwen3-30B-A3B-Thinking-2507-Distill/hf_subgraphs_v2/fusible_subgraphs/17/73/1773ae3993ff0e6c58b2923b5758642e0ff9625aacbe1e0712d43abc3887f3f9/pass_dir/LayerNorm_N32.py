"""
Pass: Replace layer_norm with normalized_shape=(32,) with a Triton kernel.
Matches both float16 and bfloat16 tiny-model graphs.
"""
import torch
from pass_dir.shared_kernels import layer_norm_triton


def pattern(in_0, in_1, in_4):
    return torch.nn.functional.layer_norm(in_4, (32,), in_1, in_0, 1e-12)


def replacement_args(in_0, in_1, in_4):
    return (in_0, in_1, in_4)


def replacement_func():
    return layer_norm_triton