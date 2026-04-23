import torch
from pass_dir.shared_fused_gelu_pad import replacement_func


def pattern(in_0):
    tmp_0 = in_0 * 0.7071067811865476
    tmp_1 = torch.erf(tmp_0)
    tmp_2 = tmp_1 + 1.0
    tmp_3 = in_0 * 0.5
    tmp_4 = tmp_3 * tmp_2
    tmp_5 = torch.ops.aten.reshape.default(tmp_4, [1, 124, 2, 768])
    tmp_6 = torch.ops.aten.reshape.default(tmp_5, [1, 248, 768])
    tmp_7 = torch.ops.aten.constant_pad_nd.default(tmp_6, [0, 0, 0, 1], 0.0)
    return tmp_7


def replacement_args(in_0):
    return (in_0, 'erf_gelu_reshape_aten_pad')