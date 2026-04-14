import torch
import triton
import triton.language as tl
from pass_dir.shared_dispatch import unified_dispatch  # shared replacement_func


# ---------------------------------------------------------------------------
# Pattern / replacement hooks
# ---------------------------------------------------------------------------

def pattern(in_6, in_5, in_2, in_1):
    """Match:  tmp = in_6 + in_5;  layer_norm(tmp, (384,), in_2, in_1, 1e-12)"""
    tmp_5 = in_6 + in_5
    tmp_6 = torch.nn.functional.layer_norm(tmp_5, (384,), in_2, in_1, 1e-12)
    return tmp_6


def replacement_args(in_6, in_5, in_2, in_1):
    # Map to unified_dispatch(a, b, c, d, route):
    #   a=in_6, b=in_5, c=in_2 (LN weight), d=in_1 (LN bias), route="add_ln"
    return (in_6, in_5, in_2, in_1, "add_ln")


def replacement_func():
    return unified_dispatch