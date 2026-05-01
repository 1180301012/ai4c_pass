import torch
from pass_dir.shared_fused_ln import fused_gelu_add_ln_dispatch


def pattern(in_2, in_3):
    tmp_2 = torch.nn.functional.gelu(in_2, approximate='none')
    tmp_3 = tmp_2.flatten(2)
    tmp_4 = tmp_3.transpose(1, 2)
    tmp_5 = tmp_4.contiguous()
    tmp_6 = in_3 + tmp_5
    tmp_7 = tmp_6.permute(0, 2, 1)
    tmp_8 = tmp_7.view(1, 256, 8, 6)
    tmp_9 = tmp_8.view(1, 256, -1)
    tmp_10 = tmp_9.permute(0, 2, 1)
    return tmp_10


def replacement_args(in_2, in_3):
    return (in_2, in_3)


def replacement_func():
    return fused_gelu_add_ln_dispatch