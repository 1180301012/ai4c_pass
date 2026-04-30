import torch
from pass_shared_impl import fused_conv_bn_avgpool_dispatch


# Literal padding=(1, 1) variant used by the resnet10t graphs.
def pattern(running_mean, running_var, bn_bias, bn_weight, conv_weight, conv_x, pool_x):
    conv2d = torch.conv2d(conv_x, conv_weight, None, (1, 1), (1, 1), (1, 1), 1)
    tmp_6 = torch.nn.functional.batch_norm(conv2d, running_mean, running_var, bn_weight, bn_bias, False, 0.1, 1e-05)
    tmp_7 = torch.nn.functional.avg_pool2d(pool_x, 2, 2, 0, True, False, None)
    return (tmp_7, tmp_6)


def replacement_args(running_mean, running_var, bn_bias, bn_weight, conv_weight, conv_x, pool_x):
    return (running_mean, running_var, bn_bias, bn_weight, conv_weight, pool_x, conv_x, 1)


def replacement_func():
    return fused_conv_bn_avgpool_dispatch