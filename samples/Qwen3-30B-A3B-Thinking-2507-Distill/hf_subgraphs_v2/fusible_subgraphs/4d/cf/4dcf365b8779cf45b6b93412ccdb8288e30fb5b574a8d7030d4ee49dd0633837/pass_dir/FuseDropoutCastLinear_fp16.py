"""
FuseDropoutCastLinear_fp16: match cast(float16) + linear for RECT_L (fp16)
Pattern: to(float16) → linear
Avoids matching dropout node which may be eliminated in decomposed graphs.
"""
import torch
from pass_dir.shared_linear_kernel import _dispatch_linear


# ── Pattern ──────────────────────────────────────────────────────────────────
# Match the cast-then-linear sequence. The dropout node (p=0.0, no-op) may
# have been eliminated by torch.compile/decomposition, so we match from `to`.

def pattern(in_0, in_1, in_2):
    to = in_2.to(torch.float16)
    linear = torch.nn.functional.linear(to, in_1, in_0)
    return linear


def replacement_args(in_0, in_1, in_2):
    # in_2 here is the pre-dropout input; dropout is a no-op so in_2 == tmp_2
    return (in_0, in_1, in_2, "rect_fp16")


def replacement_func():
    return _dispatch_linear