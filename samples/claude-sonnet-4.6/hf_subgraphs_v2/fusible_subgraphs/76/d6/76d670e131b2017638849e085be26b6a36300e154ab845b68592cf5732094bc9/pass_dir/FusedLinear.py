import torch
import sys
import os

# Import shared kernels so both passes return the SAME function object
# (required to satisfy output_pass_replacement_func_limit=1)
_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)
from shared_kernels import shared_dispatch  # noqa: E402


# ── Pattern: linear (GEMM + bias) ─────────────────────────────────────────────
# Mirrors model.py exactly:
#   torch.nn.functional.linear(in_6, in_5, in_4)
#   args: input=in_6, weight=in_5, bias=in_4

def pattern(in_6, in_5, in_4):
    out = torch.nn.functional.linear(in_6, in_5, in_4)
    return out


def replacement_args(in_6, in_5, in_4):
    # append route tag so shared_dispatch can identify this call
    return (in_6, in_5, in_4, "linear")


def replacement_func():
    return shared_dispatch