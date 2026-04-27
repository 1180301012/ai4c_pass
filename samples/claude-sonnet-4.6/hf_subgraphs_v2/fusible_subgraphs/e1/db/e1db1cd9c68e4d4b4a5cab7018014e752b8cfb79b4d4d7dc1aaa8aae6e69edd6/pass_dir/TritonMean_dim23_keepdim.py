import torch
import triton
import triton.language as tl
from pass_dir.shared_dispatch import _dispatch


# ---------------------------------------------------------------------------
# Pass: replace x.mean((2, 3), keepdim=True) with a Triton mean kernel.
# Uses the shared _dispatch function so replacement_func_limit is not hit.
# ---------------------------------------------------------------------------
def pattern(x):
    return x.mean((2, 3), keepdim=True)


def replacement_args(x):
    # Append route tag so _dispatch knows which kernel to invoke.
    return (x, "mean")


def replacement_func():
    return _dispatch