import torch
import pass_dir.shared_dispatch as _sd


def pattern(a, b):
    return torch.ops.aten.cat.default([a, b], 1)


def replacement_args(a, b):
    return (a, b, 'cat')


def replacement_func():
    return _sd.shared_dispatch