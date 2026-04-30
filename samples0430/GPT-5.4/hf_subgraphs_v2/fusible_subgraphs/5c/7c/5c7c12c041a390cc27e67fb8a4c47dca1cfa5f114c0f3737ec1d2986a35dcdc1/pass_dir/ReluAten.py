import torch
import pass_dir.shared_dispatch as _sd


def pattern(x):
    return torch.ops.aten.relu.default(x)


def replacement_args(x):
    return (x, 'relu')


def replacement_func():
    return _sd.shared_dispatch