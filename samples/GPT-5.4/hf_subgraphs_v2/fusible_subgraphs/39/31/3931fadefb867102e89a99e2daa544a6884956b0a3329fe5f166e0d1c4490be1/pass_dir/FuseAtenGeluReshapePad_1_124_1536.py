import torch
from pass_dir.shared_fused_gelu_pad import replacement_func


def pattern(in_0):
    tmp_0 = torch.ops.aten.gelu.default(in_0)
    tmp_1 = torch.ops.aten.reshape.default(tmp_0, [1, 124, 2, 768])
    tmp_2 = torch.ops.aten.reshape.default(tmp_1, [1, 248, 768])
    tmp_3 = torch.ops.aten.constant_pad_nd.default(tmp_2, [0, 0, 0, 1], 0.0)
    return tmp_3


def replacement_args(in_0):
    return (in_0, 'aten_gelu_reshape_pad')