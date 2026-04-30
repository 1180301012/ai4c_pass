import torch
from pass_dir.kernels import dispatch


def pattern(in_0, in_1, in_3, in_5, in_6):
    conv2d = torch.conv2d(in_6, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.sigmoid(conv2d)
    tmp_4 = in_5 * tmp_3
    tmp_6 = torch.cat([in_3, tmp_4], dim=1)
    tmp_11 = tmp_6.view(512, 2, 40, 32, 24)
    tmp_12 = torch.transpose(tmp_11, 1, 2)
    tmp_13 = tmp_12.contiguous()
    tmp_14 = tmp_13.view(512, 80, 32, 24)
    chunk_1 = tmp_14.chunk(2, dim=1)
    return chunk_1


def replacement_args(in_0, in_1, in_3, in_5, in_6):
    return (in_0, in_1, in_3, in_5, in_6, "p2")


def replacement_func():
    return dispatch