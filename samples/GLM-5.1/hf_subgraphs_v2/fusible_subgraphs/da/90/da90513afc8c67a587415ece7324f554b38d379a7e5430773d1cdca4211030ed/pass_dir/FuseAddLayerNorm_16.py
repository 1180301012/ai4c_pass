import torch
import triton
import triton.language as tl

from pass_dir._fused_add_ln_kernel import fused_add_layer_norm_dispatch


def pattern(bias, weight, x, y):
    tmp = x + y
    result = torch.nn.functional.layer_norm(tmp, (16,), weight, bias, 1e-05)
    return result


def replacement_args(bias, weight, x, y):
    return (bias, weight, x, y, "route_16")


def replacement_func():
    return fused_add_layer_norm_dispatch