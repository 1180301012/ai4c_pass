import torch
import triton
import triton.language as tl
from pass_dir.shared_relu_sigmoid_impl import shared_activation_dispatch


def pattern(in_0):
    tmp_1 = in_0.sigmoid()
    return tmp_1


def replacement_args(in_0):
    return (in_0, "sigmoid")


def replacement_func():
    return shared_activation_dispatch