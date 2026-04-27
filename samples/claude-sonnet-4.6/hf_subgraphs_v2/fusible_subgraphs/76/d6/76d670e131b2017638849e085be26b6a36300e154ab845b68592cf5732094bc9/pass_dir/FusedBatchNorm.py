import torch
import sys
import os

# Import shared kernels so both passes return the SAME function object
# (required to satisfy output_pass_replacement_func_limit=1)
_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)
from shared_kernels import shared_dispatch  # noqa: E402


# ── Pattern: batch_norm inference ─────────────────────────────────────────────
# Mirrors model.py exactly:
#   torch.nn.functional.batch_norm(in_7, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
#   args: input=in_7, running_mean=in_0, running_var=in_1, weight=in_3, bias=in_2

def pattern(in_7, in_0, in_1, in_3, in_2):
    out = torch.nn.functional.batch_norm(in_7, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    return out


def replacement_args(in_7, in_0, in_1, in_3, in_2):
    # append route tag so shared_dispatch can identify this call
    return (in_7, in_0, in_1, in_3, in_2, "bn")


def replacement_func():
    return shared_dispatch