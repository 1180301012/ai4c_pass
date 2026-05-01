import torch
from pass_dir.shared_kernels import shared_dispatch


def pattern(in_0, in_1, in_2):
    # in_0=bias(96,), in_1=weight(96,), in_2=input(1,16384,96)
    tmp_9 = torch.nn.functional.layer_norm(in_2, (96,), in_1, in_0, 1e-05)
    return tmp_9


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, 'ln96')


def replacement_func():
    return shared_dispatch