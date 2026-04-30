import torch
from pass_dir._shared import triton_full_dispatch


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    tmp_2 = torch.cat((in_2, in_5, in_3), dim=2)
    tmp_3 = torch.nn.functional.layer_norm(in_4, (384,), in_1, in_0, 1e-12)
    return (tmp_3, tmp_2)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5, "full_384")


def replacement_func():
    return triton_full_dispatch