import torch
from pass_dir.shared_single_output_dispatch import single_output_dispatch


def pattern(in_0, in_1, tmp_2):
    tmp_4 = torch.nn.functional.layer_norm(tmp_2, (1024,), in_1, in_0, 1e-05)
    return tmp_4


def replacement_args(in_0, in_1, tmp_2):
    return (tmp_2, in_1, in_0, 1)


def replacement_func():
    return single_output_dispatch