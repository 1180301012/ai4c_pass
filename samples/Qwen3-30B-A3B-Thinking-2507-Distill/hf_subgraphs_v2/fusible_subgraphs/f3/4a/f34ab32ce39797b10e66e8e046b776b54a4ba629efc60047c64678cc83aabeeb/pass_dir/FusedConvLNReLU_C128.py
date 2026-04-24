import torch
from pass_dir.shared_kernel import dispatch_fused_conv_ln_relu


def pattern(bias, w, ln_bias, ln_weight, x):
    conv_out = torch.conv2d(x, w, bias, (1, 1), (0, 0), (1, 1), 1)
    ln_out = torch.nn.functional.layer_norm(conv_out, (128, 1, 1), ln_weight, ln_bias, 1e-05)
    relu_out = torch.nn.functional.relu(ln_out, inplace=True)
    return relu_out


def replacement_args(bias, w, ln_bias, ln_weight, x):
    return (bias, w, ln_bias, ln_weight, x, "C128")


def replacement_func():
    return dispatch_fused_conv_ln_relu