import torch
from pass_dir.shared_fused_pool_bn_silu import shared_replacement_func


def pattern(x, weight, bias, running_mean, running_var):
    out = torch.ops.aten._native_batch_norm_legit_no_training.default(x, weight, bias, running_mean, running_var, 0.1, 1e-05)
    return out[0]


def replacement_args(x, weight, bias, running_mean, running_var):
    return (running_mean, running_var, bias, weight, x, "bn_only_static")


def replacement_func():
    return shared_replacement_func()