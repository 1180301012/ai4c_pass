import torch
from pass_dir.attn_shared import _universal


def pattern(a, b):
    """Matches: (scaled_scores + mask).softmax(-1)  for all graphs."""
    tmp_0 = a + b
    tmp_1 = torch.nn.functional.softmax(tmp_0, dim=-1)
    return tmp_1


def replacement_args(a, b):
    return (a, b)


def replacement_func():
    return _universal