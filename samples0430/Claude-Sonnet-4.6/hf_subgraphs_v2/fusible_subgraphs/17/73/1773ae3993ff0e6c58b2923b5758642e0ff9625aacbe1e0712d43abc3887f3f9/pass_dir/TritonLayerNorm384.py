"""
Pass: fuse layer_norm with normalized_shape=(384,) and eps=1e-12
Matches bfloat16 (yolos-small) and float32 (yolos-small) graphs.
Uses shared_dispatch routing so replacement_func() is identical across all passes.
"""
import torch
from pass_dir.triton_kernels import shared_dispatch


# ---------------------------------------------------------------------------
# Pattern – mirrors model.py exactly
#   torch.nn.functional.layer_norm(in_4, (384,), in_1, in_0, 1e-12)
# Free tensors in first-appearance order: in_4 (x), in_1 (weight), in_0 (bias)
# ---------------------------------------------------------------------------
def pattern(x, weight, bias):
    return torch.nn.functional.layer_norm(x, (384,), weight, bias, 1e-12)


def replacement_args(x, weight, bias):
    return (x, weight, bias, "ln384")


def replacement_func():
    return shared_dispatch