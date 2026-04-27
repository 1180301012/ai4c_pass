"""
Pass: FuseLinearDropoutTranspose_p0d0_ret34
Pattern: linear -> dropout(p=0.0, training=False) -> transpose(1,2), returns (tmp_3, tmp_4)
Matches: Graph 4 (float32, distilhubert, [1,249,512]->[1,249,768])
"""
import torch
from pass_dir._shared_linear_dispatch import _dispatch_linear_transpose, replacement_func  # noqa: F401


def pattern(in_0, in_1, in_2):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.nn.functional.dropout(linear, 0.0, False, False)
    tmp_4 = tmp_3.transpose(1, 2)
    return (tmp_3, tmp_4)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "ret34")