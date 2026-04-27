import torch
import triton
import triton.language as tl
from pass_dir.shared_kernels import fused_dispatch


# ---------------------------------------------------------------------------
# Pattern: scale * x + bias  (in_2 = relu output — relu stays native)
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2):
    tmp_3 = in_1 * in_2
    tmp_4 = tmp_3 + in_0
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    # a=bias, b=scale, c=relu_out, route="scale_add"
    return (in_0, in_1, in_2, "scale_add")


def replacement_func():
    return fused_dispatch