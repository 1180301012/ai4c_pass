import torch
from pass_dir._shared_fused_silu_add import shared_replacement_func


def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.silu(in_1, inplace=True)
    tmp_1 = tmp_0 + in_0
    return (tmp_1,)


def replacement_args(in_0, in_1):
    return (in_0, in_1, "out_only")


def replacement_func():
    return shared_replacement_func()