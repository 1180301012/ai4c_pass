import torch
from pass_dir.cgnet_shared import fused_cat_bn_prelu_pool_view


def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    conv2d = torch.conv2d(in_7, in_5, None, (1, 1), (4, 4), (4, 4), 64)
    tmp_7 = torch.cat([in_6, conv2d], 1)
    tmp_8 = torch.nn.functional.batch_norm(tmp_7, in_1, in_2, in_4, in_3, False, 0.1, 0.001)
    tmp_9 = torch.prelu(tmp_8, in_0)
    tmp_10 = torch.nn.functional.adaptive_avg_pool2d(tmp_9, 1)
    tmp_11 = tmp_10.view(128, 128)
    return (tmp_9, tmp_11)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    conv2d = torch.conv2d(in_7, in_5, None, (1, 1), (4, 4), (4, 4), 64)
    return (in_6, conv2d, in_1, in_2, in_4, in_3, in_0)


def replacement_func():
    return fused_cat_bn_prelu_pool_view