"""
Pass: FuseAddNoDropout
Match:  z = x + y; out = dropout(z, p=0.1, training=False, inplace=False)
Replace: Triton add kernel (dropout with training=False is a no-op identity).
Uses shared dispatch from shared_dispatch.py.
"""

import torch
from pass_dir.shared_dispatch import dispatch


def pattern(x, y):
    z = x + y
    out = torch.nn.functional.dropout(z, 0.1, False, False)
    return out


def replacement_args(x, y):
    return (x, y, "add_noop_dropout")


def replacement_func():
    return dispatch