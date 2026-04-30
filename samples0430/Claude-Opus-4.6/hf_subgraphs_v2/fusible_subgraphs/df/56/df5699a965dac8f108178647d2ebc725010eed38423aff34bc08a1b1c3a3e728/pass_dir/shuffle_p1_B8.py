import torch
from pass_dir.kernels import dispatch


def pattern(a, b):
    tmp_5 = torch.cat([a, b], dim=1)
    tmp_7 = tmp_5.view(8, 2, 20, 64, 48)
    tmp_8 = torch.transpose(tmp_7, 1, 2)
    tmp_9 = tmp_8.contiguous()
    tmp_10 = tmp_9.view(8, 40, 64, 48)
    chunk = tmp_10.chunk(2, dim=1)
    return chunk


def replacement_args(a, b):
    return (a, b, "p1")


def replacement_func():
    return dispatch