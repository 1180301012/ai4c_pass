import torch
from pass_dir.cca_kernels import cca_dispatch


def pattern(in_4, in_1):
    return torch.functional.einsum('bchj,bhwj->bchw', in_4, in_1)


def replacement_args(in_4, in_1):
    return (in_4, in_1, None, "einsum")


def replacement_func():
    return cca_dispatch