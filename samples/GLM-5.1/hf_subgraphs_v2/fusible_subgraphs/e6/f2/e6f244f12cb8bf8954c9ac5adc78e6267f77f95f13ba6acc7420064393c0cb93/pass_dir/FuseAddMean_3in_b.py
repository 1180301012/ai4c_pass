import torch
from pass_dir.fused_add_mean_kernel import fused_add_mean_dispatch


def pattern(in_0: torch.Tensor, in_1: torch.Tensor, in_2: torch.Tensor):
    tmp_0 = in_0 + in_1
    tmp_0 += in_2
    tmp_1 = tmp_0
    tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    return (tmp_1, tmp_2)


def replacement_args(in_0: torch.Tensor, in_1: torch.Tensor, in_2: torch.Tensor):
    return (in_0, in_1, in_2, "3in_b")


def replacement_func():
    return fused_add_mean_dispatch