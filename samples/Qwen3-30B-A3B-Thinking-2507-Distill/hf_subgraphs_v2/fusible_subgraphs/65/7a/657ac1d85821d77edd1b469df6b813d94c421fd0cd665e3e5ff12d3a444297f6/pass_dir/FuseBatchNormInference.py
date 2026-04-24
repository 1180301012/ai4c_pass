"""
Pass 2: replace batch_norm inference with an optimized Triton kernel.

Pattern has 1 output (tmp_11) to satisfy the framework's 1-output constraint.
The residual add result (tmp_10) is already in the graph from Pass 1.

replacement_args maps:
  x      -> first placeholder (matched to tmp_10)
  mean   -> second placeholder (matched to in_3 / running_mean)
  var    -> third placeholder (matched to in_4 / running_var)
  weight -> fourth placeholder (matched to in_6 / BN gamma)
  a3 dummy -> weight again (unused by "batch_norm" route, satisfies 5-arg dispatch_wrapper)
"""
import torch
from pass_dir._kernels import dispatch_wrapper


def pattern(x, running_mean, running_var, bn_weight, bn_bias):
    """
    5-input pattern for batch_norm inference (all 5 inputs are placeholders).
    """
    out = torch.nn.functional.batch_norm(
        x, running_mean, running_var, bn_weight, bn_bias, False, 0.1, 1e-05
    )
    return out   # 1 output only


def replacement_args(x, running_mean, running_var, bn_weight, bn_bias):
    # dispatch_wrapper(a0=x, a1=mean, a2=var, a3=weight, a4=bias, route="batch_norm")
    return (x, running_mean, running_var, bn_weight, bn_bias, "batch_norm")


def replacement_func():
    return dispatch_wrapper