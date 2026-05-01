"""
Pattern: torch.nn.functional.layer_norm(x, (1024,), weight, bias, 1e-05)  → single output ln
Matches the layer_norm call in H=1024 models (float32 and bfloat16).

Single-output pattern (no tuple) to maximise compatibility with the matcher.
The add/dropout nodes before layer_norm remain unchanged in the model graph;
only the layer_norm node is replaced.

Dispatch signature: shared_fused_add_ln(a, b, c, d, route)
  ln_1024 route: a=x (input), b=x (dummy repeat), c=weight, d=bias → returns ln_out
"""
import torch
import triton          # noqa: F401
import triton.language as tl  # noqa: F401

from pass_dir.shared_dispatch import shared_fused_add_ln


# ---------------------------------------------------------------------------
# Pattern: exactly mirrors model.py layer_norm call (positional args)
# ---------------------------------------------------------------------------
def pattern(x, weight, bias):
    ln = torch.nn.functional.layer_norm(x, (1024,), weight, bias, 1e-05)
    return ln


def replacement_args(x, weight, bias):
    # b slot is a dummy (x repeated); dispatch uses a=x, c=weight, d=bias
    return (x, x, weight, bias, "ln_1024")


def replacement_func():
    return shared_fused_add_ln