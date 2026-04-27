import torch
from pass_dir._shared import shared_dispatch


def pattern(tmp_4, tmp_7):
    """Match: tmp_4 * tmp_7 → single output tmp_8."""
    tmp_8 = tmp_4 * tmp_7
    return tmp_8


def replacement_args(tmp_4, tmp_7):
    return (tmp_4, tmp_7, "mul")


def replacement_func():
    return shared_dispatch