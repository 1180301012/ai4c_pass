import torch
from pass_dir.shared_kernels import dispatch_wrapper


def pattern(in_0, in_1, in_2):
    to     = in_2.to(torch.float16)
    linear = torch.nn.functional.linear(to, in_1, in_0)
    return linear


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "rect_fp16")


def replacement_func():
    return dispatch_wrapper