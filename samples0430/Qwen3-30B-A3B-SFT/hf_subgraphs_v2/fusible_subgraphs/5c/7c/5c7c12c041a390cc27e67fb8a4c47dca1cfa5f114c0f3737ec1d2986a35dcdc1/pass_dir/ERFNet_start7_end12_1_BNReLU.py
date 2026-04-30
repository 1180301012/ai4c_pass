import torch
from pass_dir.shared_bn_relu_kernel import fused_bn_relu_dispatch


def pattern(x, running_mean, running_var, weight, bias):
    return torch.nn.functional.batch_norm(
        x, running_mean, running_var, weight, bias, False, 0.1, 0.001
    )


def replacement_args(x, running_mean, running_var, weight, bias):
    return (x, running_mean, running_var, weight, bias, "r64")


def replacement_func():
    return fused_bn_relu_dispatch