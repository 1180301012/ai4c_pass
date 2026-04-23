import torch
from pass_dir.shared_fused_impl import replacement_func


def pattern(conv2d, in_2):
    tmp_2 = torch.sigmoid(conv2d)
    tmp_3 = torch.ops.aten.upsample_bilinear2d.vec(tmp_2, [64, 128], False, None)
    tmp_4 = in_2 * tmp_3
    return tmp_4


def replacement_args(conv2d, in_2):
    return (conv2d, in_2, 'sigupmul_aten')