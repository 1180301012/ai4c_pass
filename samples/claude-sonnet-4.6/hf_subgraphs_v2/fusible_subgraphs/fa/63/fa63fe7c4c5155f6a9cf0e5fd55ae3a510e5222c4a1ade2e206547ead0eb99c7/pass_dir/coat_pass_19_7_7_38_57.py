"""
Pass: D=19, Hout=7, Wout=7, a=38, b=57, C=152
Matches coat_tiny graphs.
Uses ATen ops for matmul/slice/transpose/reshape; torch.split for split.
"""
import torch
import operator
from pass_dir.coat_kernel import coat_full_replace


def pattern(in_0, in_1, in_2):
    # torch.compile/dynamo uses ATen-level ops for these:
    tmp_0 = torch.ops.aten.matmul.default(in_1, in_0)
    tmp_1 = torch.ops.aten.slice.Tensor(in_1, 2, 1, 9223372036854775807, 1)
    tmp_2 = torch.ops.aten.slice.Tensor(in_2, 2, 1, 9223372036854775807, 1)
    tmp_3 = torch.ops.aten.transpose.int(tmp_2, -1, -2)
    tmp_4 = torch.ops.aten.reshape.default(tmp_3, [1, 152, 7, 7])
    # dynamo may record torch.split (not aten.split_with_sizes):
    tmp_5 = torch.split(tmp_4, [38, 57, 57], 1)
    tmp_6 = operator.getitem(tmp_5, 0)
    tmp_7 = operator.getitem(tmp_5, 1)
    tmp_8 = operator.getitem(tmp_5, 2)
    return (tmp_0, tmp_6, tmp_7, tmp_8, tmp_1)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "19_7_7_38_57")


def replacement_func():
    return coat_full_replace