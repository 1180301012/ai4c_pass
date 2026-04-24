import torch
import triton
import triton.language as tl
from pass_dir.shared_kernel import dispatch_ln_view


# Pattern: layer_norm(x, (128,), in_1, in_0, 1e-06) → view(1, 16, 12, 128)
# Matches float16 and bfloat16 graphs with C=128, H=16, W=12
# (also matches bfloat16 start2123_end2134_97 which uses same shapes)
def pattern(in_0, in_1, tmp_10):
    tmp_11 = torch.nn.functional.layer_norm(tmp_10, (128,), in_1, in_0, 1e-06)
    tmp_12 = tmp_11.view(1, 16, 12, 128)
    return tmp_12


def replacement_args(in_0, in_1, tmp_10):
    return (in_0, in_1, tmp_10, "c128")


def replacement_func():
    return dispatch_ln_view