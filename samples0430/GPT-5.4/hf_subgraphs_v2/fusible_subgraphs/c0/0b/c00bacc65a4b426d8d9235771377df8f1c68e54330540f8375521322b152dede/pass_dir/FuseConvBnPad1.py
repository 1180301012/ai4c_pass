import torch
from pass_shared_impl import shared_dispatch


def pattern(running_mean, running_var, bn_bias, bn_weight, conv_weight, conv_x):
    conv2d = torch.conv2d(conv_x, conv_weight, None, (1, 1), (1, 1), (1, 1), 1)
    tmp_6 = torch.nn.functional.batch_norm(conv2d, running_mean, running_var, bn_weight, bn_bias, False, 0.1, 1e-05)
    return tmp_6


def replacement_args(running_mean, running_var, bn_bias, bn_weight, conv_weight, conv_x):
    return (running_mean, running_var, bn_bias, bn_weight, conv_weight, conv_x, "conv_bn_pad1")


def replacement_func():
    return shared_dispatch