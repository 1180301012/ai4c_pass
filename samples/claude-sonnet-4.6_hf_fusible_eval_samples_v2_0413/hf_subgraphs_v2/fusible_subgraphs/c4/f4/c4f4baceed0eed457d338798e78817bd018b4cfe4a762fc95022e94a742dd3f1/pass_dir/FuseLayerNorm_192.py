"""Pass: Fuse layer_norm for C=192. Single output. All dtypes."""
import torch
from pass_dir.triton_fused_dispatch import fused_dispatch


def pattern(in_0, in_1, x):
    return torch.nn.functional.layer_norm(x, (192,), in_1, in_0, 1e-05)


def replacement_args(in_0, in_1, x):
    return (x, in_1, in_0, "ln_192")


def replacement_func():
    return fused_dispatch