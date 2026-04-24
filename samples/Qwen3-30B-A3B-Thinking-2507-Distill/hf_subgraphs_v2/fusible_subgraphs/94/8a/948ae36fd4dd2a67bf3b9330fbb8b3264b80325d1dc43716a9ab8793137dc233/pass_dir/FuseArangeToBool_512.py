import torch
from pass_dir._shared_dispatch import dispatch_bool


def pattern(in_0, device):
    tmp_2 = torch.ops.aten._to_copy.default(in_0, dtype=torch.bool, device=device)
    return tmp_2


def replacement_args(in_0, device):
    return (in_0, "512")


def replacement_func():
    return dispatch_bool