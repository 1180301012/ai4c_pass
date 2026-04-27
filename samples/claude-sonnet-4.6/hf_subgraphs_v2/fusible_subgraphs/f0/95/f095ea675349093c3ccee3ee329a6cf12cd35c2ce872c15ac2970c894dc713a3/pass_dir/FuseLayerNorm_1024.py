import torch
import triton
import triton.language as tl
from pass_dir.shared_dispatch import fused_dispatch


def pattern(x, w, b):
    return torch.nn.functional.layer_norm(x, (1024,), w, b, 1e-05)


def replacement_args(x, w, b):
    return (x, w, b, "layernorm_1024")


def replacement_func():
    return fused_dispatch