"""
Pass: DepthwiseConvMean_g256_s2
Matches: depthwise conv2d (groups=256, stride=2, pad=1, k=3) followed by spatial mean
"""
import torch
from pass_dir.depthwise_conv_mean_impl import _replacement_fn_s2

def pattern(in_0, in_1):
    conv2d = torch.conv2d(in_1, in_0, None, (2, 2), (1, 1), (1, 1), 256)
    tmp_2 = conv2d.mean((2, 3), keepdim=True)
    return (conv2d, tmp_2)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

def replacement_func():
    return _replacement_fn_s2