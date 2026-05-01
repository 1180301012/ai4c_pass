import torch
from pass_dir.shared_kernels import dispatch


def pattern(bias, weight, scale_input, conv_input):
    """
    Matches: conv2d(1x1) -> sigmoid -> multiply
    Model order of args to conv2d:  in_6, in_1, in_0  →  conv_input, weight, bias
    """
    conv_out = torch.conv2d(conv_input, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    sig_out  = torch.sigmoid(conv_out)
    scaled   = scale_input * sig_out
    return scaled


def replacement_args(bias, weight, scale_input, conv_input):
    # route "css" → fused conv-sigmoid-scale
    return (bias, weight, scale_input, conv_input, "css")


def replacement_func():
    return dispatch