"""
Pass: FusedCatSliceMean_N960

Pattern:  tmp_2 = tmp_1.mean((2, 3), keepdim=True)   [single-output]
Receives tmp_1 = cat(... )[:, :, :960, :, :] as pre-computed input.
"""

import torch
from pass_dir._fused_cat_mean_kernel import shared_cat_mean_dispatch


def pattern(tmp_1):
    return tmp_1.mean((2, 3), keepdim=True)


def replacement_args(tmp_1):
    return (tmp_1, tmp_1, "N960")


def replacement_func():
    return shared_cat_mean_dispatch