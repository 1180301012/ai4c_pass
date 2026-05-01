"""
FuseLayerNorm_1024 — single-output pass for all three models.

Matches: layer_norm(x, (1024,), weight, bias, 1e-05)
Returns: single tensor tmp_10  (no tuple!)

The replacement uses smart_fused_op(*args) with 3 args (len==3 route = layernorm).
All passes share smart_fused_op, satisfying replacement_func_limit=1.

Applied AFTER FuseGeluAdd passes so that x here is the output of the
gelu+add kernel (or the original tmp_8 if gelu+add pass didn't fire).
"""

import torch
from pass_dir.smart_kernel import smart_fused_op


def pattern(x, in_1, in_0):
    tmp_10 = torch.nn.functional.layer_norm(x, (1024,), in_1, in_0, 1e-05)
    return tmp_10          # SINGLE output


def replacement_args(x, in_1, in_0):
    # in_1 = LN weight, in_0 = LN bias
    # 3 args → layernorm route in smart_fused_op
    return (x, in_1, in_0)


def replacement_func():
    return smart_fused_op