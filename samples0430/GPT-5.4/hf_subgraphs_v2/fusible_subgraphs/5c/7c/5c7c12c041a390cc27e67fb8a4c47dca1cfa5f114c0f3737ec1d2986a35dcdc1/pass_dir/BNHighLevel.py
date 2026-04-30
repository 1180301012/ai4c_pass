import torch
import pass_dir.shared_dispatch as _sd


def pattern(x, running_mean, running_var, bias, weight):
    return torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, False, 0.1, 0.001)


def replacement_args(x, running_mean, running_var, bias, weight):
    return (x, running_mean, running_var, weight, bias, 'bn')


def replacement_func():
    return _sd.shared_dispatch