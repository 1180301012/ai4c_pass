"""
FuseDropoutCastLinear_bf16: match cast(bfloat16) + linear for RECT_L (bf16)
Pattern: to(bfloat16) → linear
Avoids matching dropout node which may be eliminated in decomposed graphs.
"""
import torch
from pass_dir.shared_linear_kernel import _dispatch_linear


# ── Pattern ──────────────────────────────────────────────────────────────────
# Match the cast-then-linear sequence. The dropout node (p=0.0, no-op) may
# have been eliminated by torch.compile/decomposition, so we match from `to`.

def pattern(in_0, in_1, in_2):
    to = in_2.to(torch.bfloat16)
    linear = torch.nn.functional.linear(to, in_1, in_0)
    return linear


def replacement_args(in_0, in_1, in_2):
    # in_2 here is the pre-dropout input; dropout is a no-op so in_2 == tmp_2
    return (in_0, in_1, in_2, "rect_bf16")


def replacement_func():
    return _dispatch_linear