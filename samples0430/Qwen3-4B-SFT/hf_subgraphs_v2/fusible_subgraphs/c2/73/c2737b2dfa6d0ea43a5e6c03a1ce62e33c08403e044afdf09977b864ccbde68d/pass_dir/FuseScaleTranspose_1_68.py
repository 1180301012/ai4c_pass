import torch
from pass_dir.shared_kernel import fused_scale_transpose_1_68


# ---------------------------------------------------------------------------
# Pattern: divide by 1.6817928305074292, then transpose(-1, -2)
# ---------------------------------------------------------------------------
def pattern(in_0):
    tmp_0 = in_0 / 1.6817928305074292
    tmp_1 = tmp_0.transpose(-1, -2)
    return tmp_1


def replacement_args(in_0):
    return (in_0,)


def replacement_func():
    # Return the same function object as the 2.83 pass to satisfy
    # output_pass_replacement_func_limit.
    return fused_scale_transpose_1_68