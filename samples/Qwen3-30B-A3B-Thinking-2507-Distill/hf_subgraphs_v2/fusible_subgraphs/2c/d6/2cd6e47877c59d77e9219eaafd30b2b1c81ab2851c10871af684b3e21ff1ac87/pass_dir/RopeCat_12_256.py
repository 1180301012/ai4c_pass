import torch
from pass_dir._rope_shared import rope_dispatch


def pattern(in_2, in_3, cos_emb, sin_emb, dtype_tensor):
    tmp_1 = in_3 * cos_emb
    tmp_2 = in_3[(Ellipsis, slice(1, None, 2))]
    tmp_3 = -tmp_2
    tmp_4 = in_3[(Ellipsis, slice(None, None, 2))]
    tmp_5 = torch.stack([tmp_3, tmp_4], -1)
    tmp_6 = tmp_5.reshape((1, 12, 256, 64))
    tmp_7 = tmp_6 * sin_emb
    tmp_8 = tmp_1 + tmp_7
    tmp_9 = torch.cat([in_2, tmp_8], dim=2)
    tmp_10 = tmp_9.type_as(dtype_tensor)
    return tmp_10


def replacement_args(in_2, in_3, cos_emb, sin_emb, dtype_tensor):
    return (in_2, in_3, cos_emb, sin_emb, dtype_tensor, "cat_12_256")


def replacement_func():
    return rope_dispatch