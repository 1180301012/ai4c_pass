import torch
from pass_dir.shared_dispatch import dispatch_wrapper  # noqa: F401


def pattern(in_0, in_2):
    tmp_3 = in_2.norm(p=2, dim=-1, keepdim=True)
    tmp_4 = in_2 / tmp_3
    tmp_5 = in_0.exp()
    tmp_6 = tmp_5 * tmp_4
    return (tmp_6, tmp_4)


def replacement_args(in_0, in_2):
    return (in_0, in_2, "norm2_exp_mul")


def replacement_func():
    return dispatch_wrapper