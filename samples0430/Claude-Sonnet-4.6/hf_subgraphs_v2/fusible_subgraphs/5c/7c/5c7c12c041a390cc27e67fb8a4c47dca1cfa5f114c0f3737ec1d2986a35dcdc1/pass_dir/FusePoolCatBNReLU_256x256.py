"""
Pass: FusePoolCatBNReLU_256x256 — same 2-op pattern as other variants.
"""
import torch
from pass_dir.shared_kernel import fused_cat_bn_relu


def pattern(cat_a, cat_b, running_mean, running_var, weight, bias):
    cat = torch.cat([cat_a, cat_b], 1)
    bn = torch.nn.functional.batch_norm(cat, running_mean, running_var, weight, bias, False, 0.1, 0.001)
    return bn


def replacement_args(cat_a, cat_b, running_mean, running_var, weight, bias):
    return (cat_a, cat_b, running_mean, running_var, weight, bias)


def replacement_func():
    return fused_cat_bn_relu