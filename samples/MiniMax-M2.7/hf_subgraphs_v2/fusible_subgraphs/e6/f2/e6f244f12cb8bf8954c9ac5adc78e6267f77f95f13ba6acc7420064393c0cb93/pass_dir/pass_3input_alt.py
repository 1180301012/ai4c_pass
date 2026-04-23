"""
Pass: 3-input addition + mean (in_1 + in_2; in_2 += in_0)
Pattern: tmp_0 = in_1 + in_2; tmp_0 += in_0; tmp_1 = tmp_0; tmp_2 = tmp_1.mean((2,3), keepdim=True)
"""

from pass_dir.shared_kernels import fused_add_mean_dispatch


def pattern(in_0, in_1, in_2):
    """Pattern: in_1 + in_2; then += in_0 followed by mean((2,3), keepdim=True)"""
    tmp_0 = in_1 + in_2
    tmp_0 += in_0
    tmp_1 = tmp_0
    tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    return tmp_1, tmp_2


def replacement_args(in_0, in_1, in_2):
    """Route: 3input_10 (in_0, in_1, in_2)"""
    return (in_0, in_1, in_2, "3input_10")


def replacement_func():
    """Shared dispatch function"""
    return fused_add_mean_dispatch