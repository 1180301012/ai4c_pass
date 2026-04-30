import torch
from pass_dir.shared_softmax_expectation import replacement_impl


def pattern(in_0, in_1, in_2):
    tmp_2 = torch.nn.functional.softmax(in_2, 2)
    tmp_3 = tmp_2.reshape(-1, 17, 64, 64)
    tmp_4 = tmp_3.mul(in_0)
    tmp_5 = tmp_4.reshape(1, 17, -1)
    tmp_6 = torch.sum(tmp_5, 2, True)
    tmp_7 = tmp_3.mul(in_1)
    tmp_8 = tmp_7.reshape(1, 17, -1)
    tmp_9 = torch.sum(tmp_8, 2, True)
    tmp_10 = torch.cat([tmp_6, tmp_9], -1)
    return (tmp_3, tmp_10)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return replacement_impl