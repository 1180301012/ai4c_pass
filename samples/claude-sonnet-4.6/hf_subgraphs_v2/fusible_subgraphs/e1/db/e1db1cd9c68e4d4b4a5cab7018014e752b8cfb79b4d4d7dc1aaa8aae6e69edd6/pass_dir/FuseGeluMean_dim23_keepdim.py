import torch
import triton
import triton.language as tl
from pass_dir.shared_dispatch import _dispatch


# ---------------------------------------------------------------------------
# Pass: replace torch.nn.functional.gelu with a Triton GELU kernel.
# Uses the shared _dispatch function so replacement_func_limit is not hit.
# ---------------------------------------------------------------------------
def pattern(in_0):
    return torch.nn.functional.gelu(in_0)


def replacement_args(in_0):
    # Append route tag so _dispatch knows which kernel to invoke.
    return (in_0, "gelu")


def replacement_func():
    return _dispatch