import torch
from pass_dir.shared_single_output_dispatch import single_output_dispatch


def pattern(in_2, in_3):
    return in_2 + in_3


def replacement_args(in_2, in_3):
    return (in_2, in_3, 0)


def replacement_func():
    return single_output_dispatch