"""
Pass: FusedCatSliceMean_N672

Pattern:  tmp_2 = tmp_1.mean((2, 3), keepdim=True)   [single-output]
Receives tmp_1 = cat(... )[:, :, :672, :, :] as pre-computed input.
"""

import torch
from pass_dir._fused_cat_mean_kernel import shared_cat_mean_dispatch


def pattern(tmp_1):
    return tmp_1.mean((2, 3), keepdim=True)


def replacement_args(tmp_1):
    return (tmp_1, tmp_1, "N672")


def replacement_func():
    return shared_cat_mean_dispatch