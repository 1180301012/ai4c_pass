import torch
import triton
import triton.language as tl

from pass_dir.shared_dispatch import _fused_attn_dispatch


# ---------------------------------------------------------------
# Pattern: bmm(attn_weights, values) -> view -> transpose -> reshape
# Shapes:  tmp_2=[8,1,1], in_2=[8,1,32] -> out=[1,1,256]
#
# Key: pattern returns tmp_6 (single tensor, NOT a tuple).
# Replacement also returns a single tensor.
# Model wraps it in a tuple via its return statement.
# ---------------------------------------------------------------

def pattern(tmp_2, in_2):
    bmm_1 = torch.bmm(tmp_2, in_2)
    tmp_4 = bmm_1.view(1, 8, 1, 32)
    tmp_5 = tmp_4.transpose(1, 2)
    tmp_6 = tmp_5.reshape(1, 1, 256)
    return tmp_6


def replacement_args(tmp_2, in_2):
    return (tmp_2, in_2, "route_8h32d")


def replacement_func():
    return _fused_attn_dispatch