import torch
from pass_dir.shared_fused_impl import replacement_func


def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = torch.sigmoid(conv2d)
    tmp_3 = torch.ops.aten.upsample_bilinear2d.vec(tmp_2, [64, 128], False, None)
    tmp_4 = in_2 * tmp_3
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, 'full_aten')