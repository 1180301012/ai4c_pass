import torch
from pass_dir.cat_sigmoid_mul_kernel import triton_cat_sigmoid_multiply


def pattern(in_3, in_4, tmp_3):
    """
    Matches: cat([in_3, in_4, tmp_3], 2) -> sigmoid -> sub(0.25) -> mul(pi)
    tmp_3 is the conv2d output viewed to [N, 1, 400].
    """
    tmp_4 = torch.cat([in_3, in_4, tmp_3], 2)
    tmp_5 = tmp_4.sigmoid()
    tmp_6 = tmp_5 - 0.25
    tmp_7 = tmp_6 * 3.141592653589793
    return tmp_7


def replacement_args(in_3, in_4, tmp_3):
    return (in_3, in_4, tmp_3)


def replacement_func():
    return triton_cat_sigmoid_multiply