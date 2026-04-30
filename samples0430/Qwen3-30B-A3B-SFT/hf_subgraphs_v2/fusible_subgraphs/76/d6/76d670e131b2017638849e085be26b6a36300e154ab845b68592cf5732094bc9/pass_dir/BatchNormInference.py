import torch
from pass_dir.shared_kernels import _unified_dispatch


def pattern(x, running_mean, running_var, weight, bias):
    return torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, False, 0.1, 1e-05)


def replacement_args(x, running_mean, running_var, weight, bias):
    return (x, running_mean, running_var, weight, bias, "batch_norm")


def replacement_func():
    return _unified_dispatch