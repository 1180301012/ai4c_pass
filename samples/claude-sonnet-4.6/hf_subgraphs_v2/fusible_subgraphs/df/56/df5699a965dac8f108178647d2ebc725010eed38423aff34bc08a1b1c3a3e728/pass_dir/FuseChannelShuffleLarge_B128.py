import torch
from pass_dir.triton_kernels import dispatch


def pattern(in_3, tmp_4):
    tmp_6  = torch.cat([in_3, tmp_4], dim=1)
    tmp_11 = tmp_6.view(128, 2, 40, 32, 24)
    tmp_12 = torch.transpose(tmp_11, 1, 2)
    tmp_13 = tmp_12.contiguous()
    tmp_14 = tmp_13.view(128, 80, 32, 24)
    chunk_1  = tmp_14.chunk(2, dim=1)
    tmp_19 = chunk_1[0]
    tmp_20 = chunk_1[1]
    return tmp_19, tmp_20


def replacement_args(in_3, tmp_4):
    return (in_3, tmp_4, "shuffle")


def replacement_func():
    return dispatch