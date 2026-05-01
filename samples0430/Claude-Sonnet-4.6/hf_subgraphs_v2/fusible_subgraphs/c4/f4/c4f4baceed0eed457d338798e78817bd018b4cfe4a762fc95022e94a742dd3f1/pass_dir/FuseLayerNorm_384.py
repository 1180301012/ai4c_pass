import torch
from pass_dir.shared_kernels import shared_dispatch


def pattern(in_0, in_1, in_2):
    # in_0=bias(384,), in_1=weight(384,), in_2=input(1,1024,384)
    tmp_9 = torch.nn.functional.layer_norm(in_2, (384,), in_1, in_0, 1e-05)
    return tmp_9


def replacement_args(in_0, in_1, in_2):
    # a=in_0(bias), b=in_1(weight), c=in_2(input), route='ln384'
    return (in_0, in_1, in_2, 'ln384')


def replacement_func():
    return shared_dispatch