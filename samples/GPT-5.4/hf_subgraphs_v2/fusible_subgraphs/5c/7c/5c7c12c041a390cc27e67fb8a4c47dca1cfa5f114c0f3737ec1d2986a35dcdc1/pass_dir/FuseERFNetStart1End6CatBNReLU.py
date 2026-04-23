import torch
from pass_dir.erfnet_cat_bn_relu_shared import erfnet_pool_interp_cat_bn_relu


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    tmp_5 = torch.nn.functional.max_pool2d(in_0, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
    tmp_6 = torch.nn.functional.interpolate(tmp_5, (256, 256), None, 'bilinear', False)
    tmp_7 = torch.cat([in_5, tmp_6], 1)
    tmp_8 = torch.nn.functional.batch_norm(tmp_7, in_1, in_2, in_4, in_3, False, 0.1, 0.001)
    tmp_9 = torch.nn.functional.relu(tmp_8, inplace=False)
    return (tmp_9,)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_1, in_2, in_3, in_4, in_5, in_0, (256, 256))


def replacement_func():
    return erfnet_pool_interp_cat_bn_relu