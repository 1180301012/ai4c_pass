"""
Tuple-return pattern: torch.add(x, y) → F.layer_norm(s, (1024,), weight, bias, 1e-05)
Returns (s, ln) — tests explicit torch.add target + F.layer_norm.

If this matches, we get add+LN fusion (better speedup than LN-only).
If it fails, the LN-only fallback FuseAddDropoutLayerNorm_H1024 still applies.
"""
import torch
import triton          # noqa: F401
import triton.language as tl  # noqa: F401

from pass_dir.shared_dispatch import shared_fused_add_ln


def pattern(x, y, weight, bias):
    s  = torch.add(x, y)
    ln = torch.nn.functional.layer_norm(s, (1024,), weight, bias, 1e-05)
    return (s, ln)


def replacement_args(x, y, weight, bias):
    # a=x, b=y, c=weight, d=bias → route "add_ln_1024" returns (out, ln_out)
    return (x, y, weight, bias, "add_ln_1024")


def replacement_func():
    return shared_fused_add_ln