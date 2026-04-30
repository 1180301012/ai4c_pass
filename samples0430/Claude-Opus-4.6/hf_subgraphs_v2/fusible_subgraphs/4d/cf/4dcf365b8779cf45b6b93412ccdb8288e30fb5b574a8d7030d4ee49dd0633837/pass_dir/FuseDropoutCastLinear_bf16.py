import torch
from pass_dir.shared_impl import fused_linear_dispatch


def pattern(in_0, in_1, in_2):
    to = in_2.to(torch.bfloat16)
    linear = torch.nn.functional.linear(to, in_1, in_0)
    return (linear,)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "route_cast_bf16")


def replacement_func():
    return fused_linear_dispatch