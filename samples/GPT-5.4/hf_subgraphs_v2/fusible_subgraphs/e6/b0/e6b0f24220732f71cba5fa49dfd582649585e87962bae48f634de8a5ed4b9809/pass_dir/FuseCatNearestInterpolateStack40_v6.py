import torch
from pass_dir.shared_fused_cat_nearest_interpolate_stack40 import replacement_func

aten = torch.ops.aten


def pattern(in_0, in_1, in_2, in_3):
    tmp_0 = aten.cat.default([in_2, in_3], 1)
    tmp_1 = aten.upsample_nearest2d.vec(in_0, [40, 40], None)
    tmp_2 = aten.upsample_nearest2d.vec(in_1, [40, 40], None)
    tmp_3 = aten.stack.default([tmp_1, tmp_2, tmp_0], 0)
    return tmp_3


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)