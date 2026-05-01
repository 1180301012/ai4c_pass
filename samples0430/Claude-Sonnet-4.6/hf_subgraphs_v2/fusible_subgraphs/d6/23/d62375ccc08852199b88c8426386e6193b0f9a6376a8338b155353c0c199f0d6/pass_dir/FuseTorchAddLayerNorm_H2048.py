"""
Tuple-return pattern: torch.add(x, y) → F.layer_norm(s, (2048,), weight, bias, 1e-05)
Returns (s, ln) — H=2048 (float16 model).
"""
import torch
import triton          # noqa: F401
import triton.language as tl  # noqa: F401

from pass_dir.shared_dispatch import shared_fused_add_ln


def pattern(x, y, weight, bias):
    s  = torch.add(x, y)
    ln = torch.nn.functional.layer_norm(s, (2048,), weight, bias, 1e-05)
    return (s, ln)


def replacement_args(x, y, weight, bias):
    return (x, y, weight, bias, "add_ln_2048")


def replacement_func():
    return shared_fused_add_ln