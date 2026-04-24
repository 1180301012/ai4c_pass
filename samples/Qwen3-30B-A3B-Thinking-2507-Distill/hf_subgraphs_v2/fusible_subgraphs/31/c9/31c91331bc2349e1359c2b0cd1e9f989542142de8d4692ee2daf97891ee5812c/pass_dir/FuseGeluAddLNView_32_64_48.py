import torch
import triton
import triton.language as tl
from pass_dir.shared_kernel import dispatch_ln_view


# Pattern: layer_norm(x, (32,), in_1, in_0, 1e-06) → view(1, 64, 48, 32)
# Matches float32, float16, and bfloat16 graphs with C=32, H=64, W=48
def pattern(in_0, in_1, tmp_10):
    tmp_11 = torch.nn.functional.layer_norm(tmp_10, (32,), in_1, in_0, 1e-06)
    tmp_12 = tmp_11.view(1, 64, 48, 32)
    return tmp_12


def replacement_args(in_0, in_1, tmp_10):
    return (in_0, in_1, tmp_10, "c32")


def replacement_func():
    return dispatch_ln_view