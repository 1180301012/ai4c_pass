import torch

# Import shared dispatch wrapper to satisfy replacement_func_limit
from pass_dir.shared_dispatch import dispatch_wrapper


def pattern(arange_result):
    tmp_1 = arange_result.view(1, -1)
    tmp_2 = tmp_1.repeat(2, 1)
    return (tmp_2,)


def replacement_args(arange_result):
    return (arange_result, "partial")


def replacement_func():
    return dispatch_wrapper