import torch
import triton
import triton.language as tl
from pass_dir.shared_kernels import fused_dispatch


# ---------------------------------------------------------------------------
# Pattern: fuse max_pool2d + cat into one Triton kernel.
# in_3 = raw input for max_pool, in_4 = the other tensor concatenated.
# The cat is [max_pool_out, in_4] along dim=1.
# ---------------------------------------------------------------------------
def pattern(in_3, in_4):
    tmp_5 = torch.nn.functional.max_pool2d(in_3, 2, 1, 0, 1, ceil_mode=True, return_indices=False)
    tmp_6 = torch.cat([tmp_5, in_4], dim=1)
    return tmp_6


def replacement_args(in_3, in_4):
    # a=in3, b=in4, c=in3(dummy), route="maxpool_cat"
    return (in_3, in_4, in_3, "maxpool_cat")


def replacement_func():
    return fused_dispatch