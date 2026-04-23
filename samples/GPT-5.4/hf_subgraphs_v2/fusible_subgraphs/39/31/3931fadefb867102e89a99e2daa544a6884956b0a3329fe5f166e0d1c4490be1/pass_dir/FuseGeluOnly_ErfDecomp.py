import torch
from pass_dir.shared_fused_gelu_pad import replacement_func


def pattern(in_0):
    tmp_0 = in_0 * 0.7071067811865476
    tmp_1 = torch.erf(tmp_0)
    tmp_2 = tmp_1 + 1.0
    tmp_3 = in_0 * 0.5
    tmp_4 = tmp_3 * tmp_2
    return tmp_4


def replacement_args(in_0):
    return (in_0, 'gelu_only_erf_decomp')