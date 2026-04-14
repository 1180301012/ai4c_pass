import torch
from pass_dir.shared_dispatch import dispatch


def pattern(input, running_mean, running_var, weight, bias):
    return torch.nn.functional.batch_norm(
        input, running_mean, running_var, weight, bias, False, 0.1, 1e-05
    )


def replacement_args(input, running_mean, running_var, weight, bias):
    return (input, running_mean, running_var, weight, bias, "bn")


def replacement_func():
    return dispatch