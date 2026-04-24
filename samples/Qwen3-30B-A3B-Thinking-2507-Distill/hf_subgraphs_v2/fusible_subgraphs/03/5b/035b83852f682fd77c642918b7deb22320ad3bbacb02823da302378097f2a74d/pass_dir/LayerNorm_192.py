import torch
import triton
import triton.language as tl
from pass_dir.shared_ln_kernels import dispatch_layer_norm


def pattern(in_2, in_1, in_0):
    tmp_2 = torch.nn.functional.layer_norm(in_2, (192,), in_1, in_0, 1e-06)
    return tmp_2


def replacement_args(in_2, in_1, in_0):
    return (in_2, in_1, in_0, "ln_192")


def replacement_func():
    return dispatch_layer_norm