"""
Pass: fuse conv2d(1x1) + layer_norm(C=128,1,1). ReLU stays in graph separately.
"""
import torch
from pass_dir._kernel import fused_dispatch


def pattern(bias, weight, ln_bias, ln_weight, conv_input):
    conv_out = torch.conv2d(conv_input, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    ln_out = torch.nn.functional.layer_norm(conv_out, (128, 1, 1), ln_weight, ln_bias, 1e-05)
    return (ln_out,)


def replacement_args(bias, weight, ln_bias, ln_weight, conv_input):
    return (conv_input, weight, bias, ln_weight, ln_bias, "c128")


def replacement_func():
    return fused_dispatch