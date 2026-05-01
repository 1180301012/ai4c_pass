import torch
from pass_dir.shared_kernels import shared_dispatch


def pattern(in_2, in_3):
    tmp_2 = in_3.contiguous()
    tmp_3 = tmp_2.view(-1, 133, 133, 96)
    tmp_4 = torch.roll(tmp_3, shifts=(3, 3), dims=(1, 2))
    tmp_5 = tmp_4[(slice(None, None, None), slice(None, 128, None), slice(None, 128, None), slice(None, None, None))]
    tmp_6 = tmp_5.contiguous()
    tmp_7 = tmp_6.view(1, 16384, 96)
    tmp_8 = in_2 + tmp_7
    return tmp_8


def replacement_args(in_2, in_3):
    return (in_2, in_3, in_2, 'r133_96')


def replacement_func():
    return shared_dispatch