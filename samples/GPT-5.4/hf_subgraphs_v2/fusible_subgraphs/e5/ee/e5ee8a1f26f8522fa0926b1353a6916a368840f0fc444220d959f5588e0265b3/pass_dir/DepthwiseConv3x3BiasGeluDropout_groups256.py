import torch
from pass_dir.shared_depthwise_conv3x3_bias_gelu import replacement_func


def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (1, 1), (1, 1), 256)
    tmp_3 = torch.nn.functional.gelu(conv2d)
    tmp_4 = torch.nn.functional.dropout(tmp_3, 0.0, False, False)
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "dw256")