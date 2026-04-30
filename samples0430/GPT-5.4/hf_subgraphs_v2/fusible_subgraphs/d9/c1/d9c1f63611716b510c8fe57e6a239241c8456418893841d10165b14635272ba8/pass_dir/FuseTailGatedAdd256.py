import torch
from pass_dir.shared_kernel_update_dispatch import replacement_func


def pattern(tmp_11, tmp_12, tmp_10, tmp_13):
    tmp_14 = tmp_12.unsqueeze(-2)
    tmp_15 = tmp_11 * tmp_14
    tmp_16 = tmp_10 * tmp_13
    tmp_17 = tmp_15 + tmp_16
    return tmp_17


def replacement_args(tmp_11, tmp_12, tmp_10, tmp_13):
    return (tmp_11, tmp_12, tmp_10, tmp_13, 'tail_gated_add')