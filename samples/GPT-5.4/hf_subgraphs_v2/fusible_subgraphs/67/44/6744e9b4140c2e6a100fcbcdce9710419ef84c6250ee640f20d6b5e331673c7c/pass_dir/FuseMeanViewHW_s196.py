import torch

from pass_dir.shared_mean_view_specialized import dispatch_mean_view_specialized


def pattern(x):
    tmp_1 = x.mean((2, 3))
    tmp_4 = tmp_1.view(1, 1, -1)
    return tmp_4


def replacement_args(x):
    return (x, "s196")


def replacement_func():
    return dispatch_mean_view_specialized