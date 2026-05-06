"""
Pass: FusedCatSliceMean_N120

Pattern:  tmp_2 = tmp_1.mean((2, 3), keepdim=True)   [single-output]
Receives tmp_1 = cat([in_0,in_1],dim=1)[:,,:,120,:,:] as pre-computed input.
Replaces the mean with a faster Triton spatial-reduction kernel.
"""

import torch
from pass_dir._fused_cat_mean_kernel import shared_cat_mean_dispatch


def pattern(tmp_1):
    return tmp_1.mean((2, 3), keepdim=True)


def replacement_args(tmp_1):
    return (tmp_1, tmp_1, "N120")


def replacement_func():
    return shared_cat_mean_dispatch