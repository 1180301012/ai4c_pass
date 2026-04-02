import torch
from torch import device as _device


@torch.fx.wrap
def _fused_mask_combine(a, b):
    # a: [1, 1, N, N] bool (causal mask)
    # b: [B, N] bool (attention mask)
    # expand is a no-op; broadcasting handles it
    return a * b[slice(None, None, None), None, None, slice(None, None, None)]


def pattern(a, b):
    # a is tmp_10: [1, 1, N, N] bool from le + getitem
    # b is tmp_5: [B, N] bool from to_bool + arange_getitem
    tmp_11 = a.expand(1, -1, -1, -1)
    tmp_12 = b[slice(None, None, None), None, None, slice(None, None, None)]
    tmp_13 = tmp_11 * tmp_12
    return tmp_13


def replacement_args(a, b):
    return (a, b)


def replacement_func():
    return _fused_mask_combine