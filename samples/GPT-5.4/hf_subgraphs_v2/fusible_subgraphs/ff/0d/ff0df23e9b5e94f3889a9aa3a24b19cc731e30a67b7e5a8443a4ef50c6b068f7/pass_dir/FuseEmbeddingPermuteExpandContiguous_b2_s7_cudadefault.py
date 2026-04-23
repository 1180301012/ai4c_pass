import torch
from torch import device
from pass_dir.shared_embedding_permute_expand import fused_embedding_permute_expand_contiguous


def pattern(in_0, in_1):
    tmp_0 = in_0
    tmp_1 = in_1.to(device(type='cuda'))
    tmp_2 = torch.nn.functional.embedding(tmp_1, tmp_0, None, None, 2.0, False, False)
    tmp_3 = tmp_2.permute([2, 0, 1])
    tmp_4 = tmp_3.unsqueeze(0)
    tmp_5 = tmp_4.expand((2, -1, 7, 7))
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(in_0, in_1):
    return (in_0, in_1, 2)


def replacement_func():
    return fused_embedding_permute_expand_contiguous