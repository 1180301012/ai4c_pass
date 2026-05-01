import torch
from pass_dir.shared import dispatch_wrapper  # same object as in FuseScaleSubSplit


# ---------------------------------------------------------------------------
# Pattern: squeeze last dim + make contiguous
# Matches the two squeeze(-1)+contiguous() calls that follow split.
# ---------------------------------------------------------------------------
def pattern(x):
    t = x.squeeze(-1)
    return t.contiguous()


def replacement_args(x):
    return (x, "sq_cont")


def replacement_func():
    return dispatch_wrapper