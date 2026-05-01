"""
CombinedBNPool pass: matches batch_norm + avg_pool2d together and replaces
both with a SINGLE Triton kernel launch, halving kernel-dispatch overhead.
"""

import torch
from pass_dir.kernels import _do_combined_bn_pool


def pattern(x_bn, mean, var, weight, bias, x_pool):
    """
    Match the full BN → (tmp_6) + AvgPool → (tmp_7) subgraph.
    Both outputs are observable (returned by the model), so both are listed.
    """
    bn_out   = torch.nn.functional.batch_norm(
        x_bn, mean, var, weight, bias, False, 0.1, 1e-05)
    pool_out = torch.nn.functional.avg_pool2d(
        x_pool, 2, 2, 0, True, False, None)
    return pool_out, bn_out


def replacement_args(x_bn, mean, var, weight, bias, x_pool):
    return (x_bn, mean, var, weight, bias, x_pool)


@torch.fx.wrap
def combined_bn_pool_fn(x_bn, mean, var, weight, bias, x_pool):
    pool_out, bn_out = _do_combined_bn_pool(x_bn, mean, var, weight, bias, x_pool)
    return pool_out, bn_out


def replacement_func():
    return combined_bn_pool_fn