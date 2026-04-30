import torch
import triton
import triton.language as tl

# Import unified dispatcher from the other pass file
from pass_dir.FuseAddLayerNorm import unified_dispatcher


def pattern(ln_out, in_4, in_3):
    """
    Match pattern: slice + linear + tanh
    The original pattern from model.py:
        tmp_7 = tmp_6[(slice(None, None, None), 0)]
        linear = torch.nn.functional.linear(tmp_7, in_4, in_3)
        tmp_9 = torch.tanh(linear)
    Returns: tanh(linear) result
    """
    tmp_7 = ln_out[(slice(None, None, None), 0)]
    linear = torch.nn.functional.linear(tmp_7, in_4, in_3)
    tmp_9 = torch.tanh(linear)
    return tmp_9


def replacement_args(ln_out, in_4, in_3):
    return (ln_out, in_4, in_3, None)


def replacement_func():
    return unified_dispatcher