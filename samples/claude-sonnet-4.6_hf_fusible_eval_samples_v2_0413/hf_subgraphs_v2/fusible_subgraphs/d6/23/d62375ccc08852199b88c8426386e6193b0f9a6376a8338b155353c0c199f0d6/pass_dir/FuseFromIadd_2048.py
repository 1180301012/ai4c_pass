"""
Full fusion from iadd_result onward for hidden_dim=2048.
Covers float16/1.3B graph.
"""

import torch
from torch import device
from pass_dir.shared_fn import shared_dispatch


def pattern(in_0, in_1, iadd_result, weight, bias):
    tmp_7  = iadd_result.view(-1)
    tmp_8  = in_1.index_select(0, tmp_7)
    tmp_9  = tmp_8.view(1, 9, 2048)
    tmp_10 = tmp_9.detach()
    tmp_11 = tmp_10.to(device(type='cuda', index=0))
    tmp_12 = in_0 + tmp_11
    tmp_13 = torch.nn.functional.dropout(tmp_12, p=0.1, training=False)
    tmp_14 = torch.nn.functional.layer_norm(tmp_13, (2048,), weight, bias, 1e-05)
    return tmp_13, tmp_14


def replacement_args(in_0, in_1, iadd_result, weight, bias):
    return (in_0, in_1, iadd_result, weight, bias, "iadd_h2048")


def replacement_func():
    return shared_dispatch