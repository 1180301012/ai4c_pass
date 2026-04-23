import torch
from pass_dir.shared_routes import shared_replacement_func


def pattern(in_1, in_2):
    tmp_0 = in_1.view(-1, 1)
    tmp_1 = tmp_0 * in_2
    return tmp_1


def replacement_args(in_1, in_2):
    return (in_1, in_2, "mul")


def replacement_func():
    return shared_replacement_func()