import torch
import triton
import triton.language as tl
from pass_dir.shared_roll_only import roll_to_seq_dispatch


def pattern(in_3):
    tmp_2 = in_3.contiguous()
    tmp_3 = tmp_2.view(-1, 64, 64, 384)
    tmp_4 = torch.roll(tmp_3, shifts=(4, 4), dims=(1, 2))
    tmp_5 = tmp_4.view(1, 4096, 384)
    return tmp_5


def replacement_args(in_3):
    return (in_3, 's64_c384')


def replacement_func():
    return roll_to_seq_dispatch