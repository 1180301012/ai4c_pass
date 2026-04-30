import torch
import triton
import triton.language as tl
from pass_dir.shared_encnet_kernels import shared_replacement_func


def pattern(in_0, in_4):
    tmp_6 = in_0.view((1, 1, 32, 512))
    tmp_7 = in_4.unsqueeze(2)
    tmp_8 = tmp_7.expand((1, 4096, 32, 512))
    tmp_10 = tmp_8 - tmp_6
    return tmp_10


def replacement_args(in_0, in_4):
    return (in_0, in_4, 'broadcast_sub')


def replacement_func():
    return shared_replacement_func()