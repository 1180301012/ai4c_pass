import torch
from pass_dir.flash_attn_triton import flash_attn_wrapper


def pattern(in_0, in_1, in_2):
    tmp_0 = torch.matmul(in_0, in_1)
    tmp_1 = tmp_0 / 5.656854249492381
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.0, False, False)
    tmp_4 = torch.matmul(tmp_3, in_2)
    tmp_5 = tmp_4.permute(0, 2, 1, 3)
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "scale_5656")


def replacement_func():
    return flash_attn_wrapper