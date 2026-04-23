import torch
from pass_dir.rope_branch_common import replacement_func


def pattern(in_1, in_2, in_3, in_5, in_6):
    tmp_1 = in_3 * in_1
    tmp_2 = in_3[Ellipsis, slice(1, None, 2)]
    tmp_3 = -tmp_2
    tmp_4 = in_3[Ellipsis, slice(None, None, 2)]
    tmp_5 = torch.stack([tmp_3, tmp_4], -1)
    tmp_6 = tmp_5.reshape((1, 12, 256, 64))
    tmp_7 = tmp_6 * in_5
    tmp_8 = tmp_1 + tmp_7
    tmp_9 = torch.cat([in_2, tmp_8], dim=2)
    tmp_10 = tmp_9.type_as(in_6)
    return tmp_10


def replacement_args(in_1, in_2, in_3, in_5, in_6):
    return (in_1, in_2, in_3, in_5, in_6, "q")