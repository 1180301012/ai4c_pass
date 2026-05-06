import torch
import triton
import triton.language as tl
from pass_dir.shared_ln_relu_backend import fused_kernel


def pattern(c_in, w_conv, b_conv, ln_w, ln_b):
    """Match: conv2d(1x1, bias) -> layer_norm((16,1,1), weight, bias) -> relu"""
    conv = torch.conv2d(c_in, w_conv, b_conv, (1, 1), (0, 0), (1, 1), 1)
    ln = torch.nn.functional.layer_norm(conv, (16, 1, 1), ln_w, ln_b, 1e-05)
    relu_out = torch.nn.functional.relu(ln, inplace=True)
    return relu_out


def replacement_args(c_in, w_conv, b_conv, ln_w, ln_b):
    return (c_in, w_conv, b_conv, ln_w, ln_b, "C16")


def replacement_func():
    return fused_kernel