import torch
import triton
import triton.language as tl
from pass_dir.kernels import kernel_wrapper


def pattern(x, weight, bias):
    out = torch.nn.functional.layer_norm(x, (768,), weight, bias, 1e-05)
    return out


def replacement_args(x, weight, bias):
    return (x, None, weight, bias, "ln_768")


def replacement_func():
    return kernel_wrapper