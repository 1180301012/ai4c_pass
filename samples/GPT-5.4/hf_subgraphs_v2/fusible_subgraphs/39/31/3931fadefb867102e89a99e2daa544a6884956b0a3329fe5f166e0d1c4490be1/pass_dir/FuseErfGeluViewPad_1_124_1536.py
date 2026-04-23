import torch
from pass_dir.shared_fused_gelu_pad import replacement_func


def pattern(in_0):
    tmp_0 = in_0 * 0.7071067811865476
    tmp_1 = torch.erf(tmp_0)
    tmp_2 = tmp_1 + 1.0
    tmp_3 = in_0 * 0.5
    tmp_4 = tmp_3 * tmp_2
    tmp_5 = tmp_4.view(1, 124, 2, 768)
    tmp_6 = tmp_5.view(1, 248, 768)
    tmp_7 = torch.nn.functional.pad(tmp_6, (0, 0, 0, 1), 'constant', None)
    return tmp_7


def replacement_args(in_0):
    return (in_0, 'erf_gelu_view_pad')