import torch
from pass_dir.repvit_head_shared import repvit_head_dispatch


def pattern(in_0, in_1, in_2, in_3, in_7):
    tmp_7 = torch.nn.functional.batch_norm(in_7, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    return tmp_7


def replacement_args(in_0, in_1, in_2, in_3, in_7):
    return (in_0, in_1, in_2, in_3, in_7, "batch_norm")


def replacement_func():
    return repvit_head_dispatch