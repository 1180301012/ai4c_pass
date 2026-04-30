import torch
import pass_dir.shared_dispatch as _sd


def pattern(x, weight, bias, running_mean, running_var):
    return torch.ops.aten._native_batch_norm_legit_no_training.default(x, weight, bias, running_mean, running_var, 0.1, 0.001)[0]


def replacement_args(x, weight, bias, running_mean, running_var):
    return (x, running_mean, running_var, weight, bias, 'bn')


def replacement_func():
    return _sd.shared_dispatch