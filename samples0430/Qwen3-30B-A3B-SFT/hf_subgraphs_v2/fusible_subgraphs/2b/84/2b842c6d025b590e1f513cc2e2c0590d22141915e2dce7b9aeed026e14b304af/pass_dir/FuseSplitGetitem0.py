import torch
import triton
import triton.language as tl
from pass_dir.shared_dispatch import dispatch_wrapper


# ---------------------------------------------------------------------------
# Pattern: split_x[0].squeeze(-1).contiguous()
#   split_x is the split-result tensor  [B, S, 1] (non-contiguous view)
#   split_x is a placeholder here → no erasure conflict between passes
# ---------------------------------------------------------------------------

def pattern(split_x):
    tmp_6 = split_x.squeeze(-1)
    tmp_7 = tmp_6.contiguous()
    return tmp_7


def replacement_args(split_x):
    return (split_x, "split0")


def replacement_func():
    return dispatch_wrapper