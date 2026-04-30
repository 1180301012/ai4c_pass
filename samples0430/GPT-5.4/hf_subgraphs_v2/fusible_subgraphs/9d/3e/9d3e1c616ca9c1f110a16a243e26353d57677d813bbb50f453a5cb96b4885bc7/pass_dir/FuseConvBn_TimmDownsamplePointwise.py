import torch
from pass_dir.shared_pointwise_conv_bn import replacement_func


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    conv2d = torch.conv2d(in_5, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_6 = torch.nn.functional.batch_norm(conv2d, in_1, in_2, in_4, in_3, False, 0.1, 1e-05)
    return tmp_6


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_5, in_0, in_1, in_2, in_4, in_3)