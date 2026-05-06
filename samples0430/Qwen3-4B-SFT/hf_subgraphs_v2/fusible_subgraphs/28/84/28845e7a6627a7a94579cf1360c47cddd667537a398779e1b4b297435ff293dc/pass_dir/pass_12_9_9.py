"""
Pass: fuse add + max_fill(-inf) + view(12,9,9) + softmax + dropout(training=False)
for the bfloat16 and float16 graphs with attention row shape (12, 9, 9).
"""
import torch
from torch import device
from pass_dir.shared_kernel import dispatch_fused_add_mask_softmax


def pattern(in_0, in_1):
    tmp_0 = in_1 + in_0
    tmp_1 = torch.tensor(-3.4028234663852886e+38, device=device(type='cuda', index=0))
    tmp_2 = torch.max(tmp_0, tmp_1)
    tmp_3 = tmp_2.view(12, 9, 9)
    tmp_4 = torch.nn.functional.softmax(tmp_3, dim=-1)
    tmp_5 = torch.nn.functional.dropout(tmp_4, p=0.1, training=False)
    return (tmp_5,)


def replacement_args(in_0, in_1):
    return (in_0, in_1, "route_12_9_9")


def replacement_func():
    return dispatch_fused_add_mask_softmax