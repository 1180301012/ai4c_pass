"""
Pass: 2-input addition + mean (in_0 + in_1)
Pattern: tmp_0 = in_0 + in_1; tmp_1 = tmp_0; tmp_2 = tmp_1.mean((2,3), keepdim=True)
"""

from pass_dir.shared_kernels import fused_add_mean_dispatch


def pattern(in_0, in_1):
    """Pattern: in_0 + in_1 followed by mean((2,3), keepdim=True)"""
    tmp_0 = in_0 + in_1
    tmp_1 = tmp_0
    tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    return tmp_1, tmp_2


def replacement_args(in_0, in_1):
    """Route: 2input_01 (in_0, in_1)"""
    return (in_0, in_1, "2input_01")


def replacement_func():
    """Shared dispatch function"""
    return fused_add_mean_dispatch