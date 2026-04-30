import torch
from pass_dir.shared_tail_fusion import fused_tail_dispatch


def pattern(x, running_mean, running_var, bn_bias, bn_weight):
    out = torch.nn.functional.batch_norm(x, running_mean, running_var, bn_weight, bn_bias, False, 0.1, 1e-05)
    return out


def replacement_args(x, running_mean, running_var, bn_bias, bn_weight):
    zeros = bn_weight * 0.0
    ones = zeros + 1.0
    bn_scale_1d = bn_weight * (running_var + 1e-05).rsqrt()
    bn_bias_1d = bn_bias - running_mean * bn_scale_1d
    return (x, ones, bn_scale_1d, bn_bias_1d, zeros, "bn_only")


def replacement_func():
    return fused_tail_dispatch