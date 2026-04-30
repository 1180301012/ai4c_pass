import torch
import triton
import triton.language as tl
from pass_dir.shared_dispatch import dispatch_wrapper


# ---------------------------------------------------------------------------
# Pattern: split_x[1].squeeze(-1).contiguous()  — method-call form
# ---------------------------------------------------------------------------

def pattern(split_x):
    tmp_8 = split_x.squeeze(-1)
    tmp_9 = tmp_8.contiguous()
    return tmp_9


def replacement_args(split_x):
    return (split_x, "split1")


def replacement_func():
    return dispatch_wrapper