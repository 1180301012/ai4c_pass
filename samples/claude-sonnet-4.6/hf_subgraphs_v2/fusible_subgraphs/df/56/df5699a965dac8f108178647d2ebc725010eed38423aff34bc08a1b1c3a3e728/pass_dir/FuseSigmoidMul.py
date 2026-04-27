import torch
from pass_dir.triton_kernels import dispatch


def pattern(conv2d_out, in_5):
    tmp_3 = torch.sigmoid(conv2d_out)
    tmp_4 = in_5 * tmp_3
    return tmp_4


def replacement_args(conv2d_out, in_5):
    return (conv2d_out, in_5, "sigmoid_mul")


def replacement_func():
    return dispatch