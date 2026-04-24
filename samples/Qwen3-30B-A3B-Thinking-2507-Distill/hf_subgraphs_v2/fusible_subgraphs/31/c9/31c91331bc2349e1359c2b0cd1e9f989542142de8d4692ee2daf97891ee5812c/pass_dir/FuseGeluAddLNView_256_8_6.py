import torch
import triton
import triton.language as tl
from pass_dir.shared_kernel import dispatch_ln_view


# Pattern: layer_norm(x, (256,), in_1, in_0, 1e-06) → view(1, 8, 6, 256)
# Matches float16 and bfloat16 graphs with C=256, H=8, W=6
def pattern(in_0, in_1, tmp_10):
    tmp_11 = torch.nn.functional.layer_norm(tmp_10, (256,), in_1, in_0, 1e-06)
    tmp_12 = tmp_11.view(1, 8, 6, 256)
    return tmp_12


def replacement_args(in_0, in_1, tmp_10):
    return (in_0, in_1, tmp_10, "c256")


def replacement_func():
    return dispatch_ln_view