"""
Pass: ATen-level pattern for max-fill(-inf) + view(12,9,9) + softmax + dropout
Uses torch.ops.aten ops that appear in the Dynamo-compiled graph.
"""
import torch
from pass_dir.shared_kernel import dispatch_fused_max_fill_softmax


def pattern(x, y):
    tmp_2 = torch.ops.aten.max.dim(x, y, -1)
    tmp_3 = torch.ops.aten.view.default(tmp_2, [12, 9, 9])
    tmp_4 = torch.ops.aten._softmax.default(tmp_3, -1, False)
    tmp_5 = torch.ops.aten.dropout.default(tmp_4, 0.1, False)
    return (tmp_5,)


def replacement_args(x, y):
    return (x, y, "route_12_9_9")


def replacement_func():
    return dispatch_fused_max_fill_softmax