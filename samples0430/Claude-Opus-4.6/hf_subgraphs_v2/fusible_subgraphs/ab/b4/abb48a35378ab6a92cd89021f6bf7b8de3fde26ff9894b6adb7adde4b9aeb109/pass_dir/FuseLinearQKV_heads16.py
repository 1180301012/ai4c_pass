import torch
from pass_dir.shared_kernel import fused_linear_qkv_dispatch


def pattern(weight, x):
    linear = torch.nn.functional.linear(x, weight, None)
    reshaped = linear.reshape(1, 197, 3, 16, 48)
    permuted = reshaped.permute(2, 0, 3, 1, 4)
    unbound = permuted.unbind(0)
    return unbound


def replacement_args(weight, x):
    return (weight, x)


def replacement_func():
    return fused_linear_qkv_dispatch