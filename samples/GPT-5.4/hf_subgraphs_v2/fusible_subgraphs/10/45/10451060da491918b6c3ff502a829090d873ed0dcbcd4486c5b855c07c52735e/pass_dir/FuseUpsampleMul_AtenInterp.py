import torch
from pass_dir.shared_fused_impl import replacement_func


def pattern(tmp_2, in_2):
    tmp_3 = torch.ops.aten.upsample_bilinear2d.vec(tmp_2, [64, 128], False, None)
    tmp_4 = in_2 * tmp_3
    return tmp_4


def replacement_args(tmp_2, in_2):
    return (tmp_2, in_2, 'upmul_aten')