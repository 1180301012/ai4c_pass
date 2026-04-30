import torch
from pass_dir.shared_fused_pool_cat_bn_relu import fused_pool_cat_bn_relu_dispatch


def pattern(running_mean, running_var, bias, weight, cat_other, pool_src):
    tmp_0 = torch.nn.functional.max_pool2d(pool_src, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
    tmp_1 = torch.nn.functional.interpolate(tmp_0, (64, 64), None, 'bilinear', False)
    tmp_2 = torch.cat([cat_other, tmp_1], 1)
    tmp_3 = torch.nn.functional.batch_norm(tmp_2, running_mean, running_var, weight, bias, False, 0.1, 0.001)
    tmp_4 = torch.nn.functional.relu(tmp_3, inplace=False)
    return (tmp_4,)


def replacement_args(running_mean, running_var, bias, weight, cat_other, pool_src):
    return (pool_src, cat_other, running_mean, running_var, bias, weight, 'size64')


def replacement_func():
    return fused_pool_cat_bn_relu_dispatch