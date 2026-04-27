import torch
from pass_dir.triton_kernels import dispatch


def pattern(in_2, in_4):
    tmp_5  = torch.cat([in_2, in_4], dim=1)
    tmp_7  = tmp_5.view(8, 2, 20, 64, 48)
    tmp_8  = torch.transpose(tmp_7, 1, 2)
    tmp_9  = tmp_8.contiguous()
    tmp_10 = tmp_9.view(8, 40, 64, 48)
    chunk  = tmp_10.chunk(2, dim=1)
    tmp_16 = chunk[0]
    tmp_17 = chunk[1]
    return tmp_16, tmp_17


def replacement_args(in_2, in_4):
    return (in_2, in_4, "shuffle")


def replacement_func():
    return dispatch