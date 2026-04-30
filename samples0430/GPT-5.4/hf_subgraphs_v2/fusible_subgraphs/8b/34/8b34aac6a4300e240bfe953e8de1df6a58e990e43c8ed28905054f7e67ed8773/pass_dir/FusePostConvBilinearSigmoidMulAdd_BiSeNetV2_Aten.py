import torch
from pass_dir.shared_bisenetv2_postconv_kernels import shared_dispatch


def pattern(tmp_2, in_2, in_3, in_4):
    tmp_3 = torch.ops.aten.upsample_bilinear2d.vec(in_4, [64, 64], False, None)
    tmp_4 = torch.ops.aten.sigmoid.default(tmp_3)
    tmp_5 = torch.ops.aten.mul.Tensor(in_3, tmp_4)
    tmp_6 = torch.ops.aten.sigmoid.default(tmp_2)
    tmp_7 = torch.ops.aten.mul.Tensor(in_2, tmp_6)
    tmp_8 = torch.ops.aten.upsample_bilinear2d.vec(tmp_7, [64, 64], False, None)
    tmp_9 = torch.ops.aten.add.Tensor(tmp_5, tmp_8)
    return tmp_9


def replacement_args(tmp_2, in_2, in_3, in_4):
    return (tmp_2, in_2, in_3, in_4, 'full')


def replacement_func():
    return shared_dispatch