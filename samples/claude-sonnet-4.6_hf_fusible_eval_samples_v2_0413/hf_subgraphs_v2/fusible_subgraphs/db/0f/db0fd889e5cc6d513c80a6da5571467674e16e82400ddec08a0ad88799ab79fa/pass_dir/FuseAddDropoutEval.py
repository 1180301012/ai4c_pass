"""
Pass: FuseAddDropoutEval

Matches the add + dropout (eval-mode, training=False) pattern:
  tmp_23 = tmp_12 + tmp_22          [1, 236, 32]
  tmp_24 = dropout(tmp_23, 0.1, False, False)  -- identity in eval

Uses shared routing dispatch.
"""

import torch
from pass_dir.yolos_shared import yolos_dispatch


def pattern(x, y):
    z = x + y
    out = torch.nn.functional.dropout(z, 0.1, False, False)
    return out


def replacement_args(x, y):
    return (x, y, "add_dropout")


def replacement_func():
    return yolos_dispatch