import torch
from pass_dir.shared_kernel import fused_conv_mean_stride1_g384


def pattern(x, w):
    conv_out = torch.conv2d(x, w, None, (1, 1), (1, 1), (1, 1), 384)
    mean_out = conv_out.mean((2, 3), keepdim=True)
    return (conv_out, mean_out)


def replacement_args(x, w):
    return (x, w)


def replacement_func():
    return fused_conv_mean_stride1_g384