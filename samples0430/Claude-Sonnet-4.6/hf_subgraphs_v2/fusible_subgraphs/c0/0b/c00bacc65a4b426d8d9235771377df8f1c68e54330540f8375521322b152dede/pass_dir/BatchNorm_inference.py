import torch
from pass_dir.kernels import fused_dispatch


def pattern(x, running_mean, running_var, weight, bias):
    return torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, False, 0.1, 1e-05)


def replacement_args(x, running_mean, running_var, weight, bias):
    # Route "bn" → fused_dispatch("bn", x, mean, var, weight, bias)
    return ("bn", x, running_mean, running_var, weight, bias)


def replacement_func():
    return fused_dispatch