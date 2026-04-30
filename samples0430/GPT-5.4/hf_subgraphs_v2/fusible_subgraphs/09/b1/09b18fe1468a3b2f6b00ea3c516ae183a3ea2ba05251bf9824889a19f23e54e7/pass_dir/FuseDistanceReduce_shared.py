import torch
import triton
import triton.language as tl
from pass_dir.shared_encnet_kernels import shared_replacement_func


def pattern(in_1, in_2):
    tmp_1 = in_1 - in_2
    tmp_2 = tmp_1.pow(2)
    tmp_3 = tmp_2.sum(dim = 3)
    return tmp_3


def replacement_args(in_1, in_2):
    return (in_1, in_2, 'distance_reduce')


def replacement_func():
    return shared_replacement_func()