"""
FuseGeluAdd_drop005 — single-output pass for bfloat16 model (dropout=0.05).

Matches: gelu → transpose(1,2) → add → dropout(0.05, inference)
Returns: single tensor tmp_8  (no tuple!)

The replacement uses smart_fused_op(*args) with 2 args (len==2 route = gelu+add).
All passes share smart_fused_op, satisfying replacement_func_limit=1.
"""

import torch
from pass_dir.smart_kernel import smart_fused_op


def pattern(tmp_4, in_3):
    tmp_5 = torch.nn.functional.gelu(tmp_4)
    tmp_6 = tmp_5.transpose(1, 2)
    tmp_7 = in_3 + tmp_6
    tmp_8 = torch.nn.functional.dropout(tmp_7, 0.05, False, False)
    return tmp_8           # SINGLE output


def replacement_args(tmp_4, in_3):
    return (tmp_4, in_3)  # 2 args → gelu+add route in smart_fused_op


def replacement_func():
    return smart_fused_op