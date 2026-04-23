import torch

from pass_dir.shared_inplace_silu_mean_view import shared_fused_inplace_silu_mean_view_dispatch


def pattern(in_0: torch.Tensor, in_1):
    tmp_0 = torch.nn.functional.silu(in_1, inplace=True)
    tmp_1 = tmp_0.mean((2, 3))
    tmp_2 = in_0 // 8
    tmp_3 = torch.sym_sum([1, tmp_2])
    tmp_4 = tmp_1.view(1, 1, -1)
    return (tmp_0, tmp_4, tmp_3)


def replacement_args(in_0, in_1):
    return (in_0, in_1, "three_out")


def replacement_func():
    return shared_fused_inplace_silu_mean_view_dispatch