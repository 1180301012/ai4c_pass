"""
Pass: FuseLinearDropoutTranspose_p0d0_ret43
Pattern: linear -> dropout(p=0.0, training=False)  →  fast Triton linear
Returns single tensor tmp_3; transpose (tmp_4=tmp_3.T) stays in graph as free view.
Matches: Graph 3 (float16, tiny-UniSpeechSat), Graph 4 (float32, distilhubert),
         Graph 5 (bfloat16, tiny-UniSpeechSat)  — all have p=0.0 dropout.
"""
import torch
from pass_dir._shared_linear_dispatch import _compute_linear_bias, replacement_func  # noqa: F401


def pattern(in_0, in_1, in_2):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.nn.functional.dropout(linear, 0.0, False, False)
    return tmp_3


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)