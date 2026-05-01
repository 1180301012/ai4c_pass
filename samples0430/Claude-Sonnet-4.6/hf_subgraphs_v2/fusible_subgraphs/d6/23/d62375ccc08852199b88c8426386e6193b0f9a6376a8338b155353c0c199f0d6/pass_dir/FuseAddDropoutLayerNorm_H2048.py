"""
Pattern: torch.nn.functional.layer_norm(x, (2048,), weight, bias, 1e-05)  → single output ln
Matches the layer_norm call in H=2048 model (float16).

Single-output pattern (no tuple) to maximise compatibility with the matcher.
"""
import torch
import triton          # noqa: F401
import triton.language as tl  # noqa: F401

from pass_dir.shared_dispatch import shared_fused_add_ln


# ---------------------------------------------------------------------------
# Pattern: exactly mirrors model.py layer_norm call (positional args)
# ---------------------------------------------------------------------------
def pattern(x, weight, bias):
    ln = torch.nn.functional.layer_norm(x, (2048,), weight, bias, 1e-05)
    return ln


def replacement_args(x, weight, bias):
    # b slot is a dummy (x repeated); dispatch uses a=x, c=weight, d=bias
    return (x, x, weight, bias, "ln_2048")


def replacement_func():
    return shared_fused_add_ln