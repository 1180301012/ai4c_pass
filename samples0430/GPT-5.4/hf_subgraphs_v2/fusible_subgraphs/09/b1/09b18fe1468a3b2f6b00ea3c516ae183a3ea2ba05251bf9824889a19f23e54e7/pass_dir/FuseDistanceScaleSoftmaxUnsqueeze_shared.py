import torch
import triton
import triton.language as tl
from pass_dir.shared_encnet_kernels import shared_replacement_func


def pattern(in_1, in_2, in_3):
    tmp_1 = in_1 - in_2
    tmp_2 = tmp_1.pow(2)
    tmp_3 = tmp_2.sum(dim = 3)
    tmp_4 = in_3 * tmp_3
    tmp_5 = torch.nn.functional.softmax(tmp_4, dim = 2)
    tmp_9 = tmp_5.unsqueeze(3)
    return tmp_9


def replacement_args(in_1, in_2, in_3):
    return (in_1, in_2, in_3, 'softmax_path')


def replacement_func():
    return shared_replacement_func()