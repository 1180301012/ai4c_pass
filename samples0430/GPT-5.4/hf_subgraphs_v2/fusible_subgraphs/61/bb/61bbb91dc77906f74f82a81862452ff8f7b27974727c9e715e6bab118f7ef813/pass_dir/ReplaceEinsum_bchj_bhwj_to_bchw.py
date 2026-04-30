import torch
from pass_dir.shared_ccnet_kernels import shared_replacement_func


def pattern(in_4, in_1):
    einsum = torch.functional.einsum('bchj,bhwj->bchw', in_4, in_1)
    return einsum


def replacement_args(in_4, in_1):
    return (in_4, in_1, 'einsum')


def replacement_func():
    return shared_replacement_func()