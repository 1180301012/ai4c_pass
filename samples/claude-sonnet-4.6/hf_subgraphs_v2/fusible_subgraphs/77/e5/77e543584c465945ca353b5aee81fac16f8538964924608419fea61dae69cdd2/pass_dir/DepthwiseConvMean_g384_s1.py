"""
Pass: DepthwiseConvMean_g384_s1 (repurposed as FastMeanHW)
Single-output pattern: replaces x.mean((2,3), keepdim=True) with a fast
Triton reduction kernel.  Single output avoids FX multi-output crash.
"""
import torch
from pass_dir.depthwise_conv_mean_impl import fast_mean_hw


def pattern(x):
    return x.mean((2, 3), keepdim=True)


def replacement_args(x):
    return (x,)


def replacement_func():
    return fast_mean_hw