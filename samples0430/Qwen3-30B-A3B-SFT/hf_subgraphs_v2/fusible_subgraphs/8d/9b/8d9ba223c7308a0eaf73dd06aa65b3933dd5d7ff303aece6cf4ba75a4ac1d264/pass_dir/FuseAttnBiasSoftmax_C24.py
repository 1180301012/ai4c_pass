import torch
import triton
import triton.language as tl
from pass_dir._shared_dispatch import dispatch_wrapper


# ── Pattern: sigmoid → mul(16) → unsqueeze(0) ─────────────────────────────────
def pattern(tmp_8):
    tmp_9  = torch.sigmoid(tmp_8)
    tmp_10 = 16 * tmp_9
    tmp_11 = tmp_10.unsqueeze(0)
    return tmp_11


def replacement_args(tmp_8):
    return (tmp_8, "sigmoid")


def replacement_func():
    return dispatch_wrapper