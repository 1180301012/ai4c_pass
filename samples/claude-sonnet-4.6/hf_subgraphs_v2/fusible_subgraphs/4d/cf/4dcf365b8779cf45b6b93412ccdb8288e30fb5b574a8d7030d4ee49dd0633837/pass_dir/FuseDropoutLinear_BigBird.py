import torch
from pass_dir.shared_kernels import dispatch_wrapper


def pattern(in_0, in_1, in_2):
    tmp_3  = torch.nn.functional.dropout(in_2, 0.1, False, False)
    linear = torch.nn.functional.linear(tmp_3, in_1, in_0)
    return linear


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "bigbird")


def replacement_func():
    return dispatch_wrapper