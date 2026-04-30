import torch
import triton
import triton.language as tl
from pass_dir.shared_layernorm_kernel import triton_layer_norm


def pattern(bias, weight, x):
    result = torch.nn.functional.layer_norm(x, (16,), weight, bias, 1e-05)
    return result


def replacement_args(bias, weight, x):
    return (bias, weight, x)


def replacement_func():
    return triton_layer_norm