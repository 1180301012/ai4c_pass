"""
Pass: FuseLinearDropoutTranspose_return_A

Matches linear + dropout(p=0.1, training=False) → returns the dropout output.
The downstream transpose(1,2) node is LEFT IN THE GRAPH (outside the matched subgraph)
and applied to the replacement output automatically.
"""
import torch
from pass_dir.linear_transpose_kernel import fused_linear_bias


def pattern(bias, weight, x):
    linear = torch.nn.functional.linear(x, weight, bias)
    out = torch.nn.functional.dropout(linear, 0.1, False, False)
    return out


def replacement_args(bias, weight, x):
    # Return shape values as FX proxies (not ints) — the framework traces these
    # and records them as graph nodes.  No int() wrapping.
    return (bias, weight, x, weight.shape[0], x.shape[2], x.shape[0], x.shape[1])


def replacement_func():
    return fused_linear_bias