import torch
from pass_dir._shared_fused_silu_add import shared_replacement_func


def pattern(in_1):
    tmp_0 = torch.nn.functional.silu(in_1, inplace=True)
    return tmp_0


def replacement_args(in_1):
    return (in_1, in_1, "silu_only")


def replacement_func():
    return shared_replacement_func()