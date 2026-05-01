import torch
from pass_dir.shared_kernel import fused_conv_ln_relu


def pattern(in_0, in_1, in_2, in_3, in_4):
    """
    Match: conv2d(1x1) -> layer_norm(normalized_shape=(16,1,1)) -> relu
    in_0: conv_bias   [16]
    in_1: conv_weight [16, C_in, 1, 1]
    in_2: ln_bias     [16, 1, 1]
    in_3: ln_weight   [16, 1, 1]
    in_4: input       [N, C_in, 1, 1]
    """
    conv2d = torch.conv2d(in_4, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_5 = torch.nn.functional.layer_norm(conv2d, (16, 1, 1), in_3, in_2, 1e-05)
    tmp_6 = torch.nn.functional.relu(tmp_5, inplace=True)
    return tmp_6


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


def replacement_func():
    return fused_conv_ln_relu