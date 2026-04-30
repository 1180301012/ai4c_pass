import torch
import triton
import triton.language as tl
from pass_dir.shared_arange_repeat import dispatch_view_repeat


# ---------------------------------------------------------------------------
# Unified pattern: x.view(1, -1) → repeat(2, 1)
# Works for any N (128 for RECT_L, 1000 for GAE).
# Placeholder x binds to the arange output; dispatch_view_repeat
# reads x.shape[0] at compile/runtime to select the right kernel.
# ---------------------------------------------------------------------------

def pattern(x):
    tmp_1 = x.view(1, -1)
    tmp_2 = tmp_1.repeat(2, 1)
    return tmp_2


def replacement_args(x):
    return (x,)


def replacement_func():
    return dispatch_view_repeat