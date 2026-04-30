"""
Pass: FuseDropoutLinear
Matches: dropout(x, p, training=False) -> linear(out, weight, bias)
         where dropout is an identity (training=False)
         and there is NO dtype conversion before linear.

Covers: bigbird-roberta-base graphs (bfloat16 and float16)
Uses shared_dispatch with route="bigbird" for unified replacement_func.
"""

import torch
from pass_dir.shared_kernels import shared_dispatch


def pattern(bias, weight, x):
    """
    Match dropout(x, 0.1, False, False) -> linear(out, weight, bias)
    Exact positional args as in bigbird-roberta-base model.py.
    """
    tmp = torch.nn.functional.dropout(x, 0.1, False, False)
    linear = torch.nn.functional.linear(tmp, weight, bias)
    return linear


def replacement_args(bias, weight, x):
    # Append route tag so shared_dispatch knows to use float16 kernel
    return (bias, weight, x, "bigbird")


def replacement_func():
    return shared_dispatch