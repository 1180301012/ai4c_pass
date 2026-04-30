import torch
import pass_dir.shared_dispatch as _sd


def pattern(x):
    return torch.nn.functional.relu(x, inplace=False)


def replacement_args(x):
    return (x, 'relu')


def replacement_func():
    return _sd.shared_dispatch