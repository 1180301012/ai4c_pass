"""
Pass: 1-input (identity) + mean (0 + in_0; in_0 += 0)
Pattern: tmp_0 = 0 + in_0; tmp_0 += 0; tmp_1 = tmp_0; tmp_2 = tmp_1.mean((2,3), keepdim=True)
"""

from pass_dir.shared_kernels import fused_add_mean_dispatch


def pattern(in_0):
    """Pattern: 0 + in_0; in_0 += 0 followed by mean((2,3), keepdim=True)
    
    This is equivalent to identity, but we still need to match the pattern exactly.
    """
    tmp_0 = 0 + in_0
    tmp_0 += 0
    tmp_1 = tmp_0
    tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    return tmp_1, tmp_2


def replacement_args(in_0):
    """Route: 1input (in_0)"""
    return (in_0, "1input")


def replacement_func():
    """Shared dispatch function"""
    return fused_add_mean_dispatch