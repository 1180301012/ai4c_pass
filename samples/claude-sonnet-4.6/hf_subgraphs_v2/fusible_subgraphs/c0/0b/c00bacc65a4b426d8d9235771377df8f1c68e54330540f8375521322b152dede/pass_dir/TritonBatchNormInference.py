import torch
from pass_dir.shared_kernels import dispatch_bn_pool


def pattern(x, running_mean, running_var, weight, bias):
    return torch.nn.functional.batch_norm(
        x, running_mean, running_var, weight, bias, False, 0.1, 1e-05
    )


def replacement_args(x, running_mean, running_var, weight, bias):
    # Append route tag so dispatch_bn_pool knows which branch to execute
    return (x, running_mean, running_var, weight, bias, "bn")


def replacement_func():
    return dispatch_bn_pool