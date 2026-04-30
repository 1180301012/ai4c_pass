import torch
from pass_dir.shared_kernel import replacement_func


def pattern(in_0, in_1):
    tmp_0 = torch.cat([in_0, in_1], dim=1)
    tmp_1 = tmp_0[:, :960, :, :]
    return tmp_1


def replacement_args(in_0, in_1):
    return (in_0, in_1, "route_960")