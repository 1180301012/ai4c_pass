"""
Full fusion from iadd_result (operator.iadd output = call_function) onward.
Pattern: iadd_result.view(-1) → index_select → view(1,9,1024) → detach
         → to(device) → in_0 + pos → dropout → layer_norm
Replaces with a single fused Triton kernel (index lookup + add + LN).
Returns (tmp_13, tmp_14) matching the model's return.
Covers float32 and bfloat16 graphs with hidden_dim=1024.
"""

import torch
from torch import device
from pass_dir.shared_fn import shared_dispatch


def pattern(in_0, in_1, iadd_result, weight, bias):
    tmp_7  = iadd_result.view(-1)
    tmp_8  = in_1.index_select(0, tmp_7)
    tmp_9  = tmp_8.view(1, 9, 1024)
    tmp_10 = tmp_9.detach()
    tmp_11 = tmp_10.to(device(type='cuda', index=0))
    tmp_12 = in_0 + tmp_11
    tmp_13 = torch.nn.functional.dropout(tmp_12, p=0.1, training=False)
    tmp_14 = torch.nn.functional.layer_norm(tmp_13, (1024,), weight, bias, 1e-05)
    return tmp_13, tmp_14


def replacement_args(in_0, in_1, iadd_result, weight, bias):
    # a=in_0, b=in_1, c=iadd_result, d=weight, e=bias
    return (in_0, in_1, iadd_result, weight, bias, "iadd_h1024")


def replacement_func():
    return shared_dispatch