import torch
from pass_dir.shared_weighted_expand import fused_dispatch


# ---------------------------------------------------------------------------
# Pattern: matches float16 graph with new_zeros((128, 128))
# Mirrors model.py exactly for RECT_L graph (N=256, F=128)
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2):
    tmp_0 = in_1.view(-1, 1)
    tmp_1 = tmp_0 * in_2
    tmp_2 = in_0.view((-1, 1))
    tmp_3 = tmp_2.expand_as(tmp_1)
    tmp_4 = tmp_1.new_zeros((128, 128))
    return (tmp_3, tmp_4, tmp_1)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "route_128")


def replacement_func():
    # Returns the SAME fused_dispatch object as FuseWeightedExpand_1000_16,
    # satisfying the output_pass_replacement_func_limit constraint.
    return fused_dispatch