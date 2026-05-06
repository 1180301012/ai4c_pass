"""
Pass: PairDiff_128
Matches the pair_diff pattern for the 128-channel variant (used in fp32 graphs).

Pattern matched on the 19×19=361, 7×7=49 post-reshape shape.
"""
import torch
import triton
import triton.language as tl
from pass_dir.pair_diff_kernel import pair_diff_wrapper, dispatch_wrapper


def pattern(t, m):
    return t.masked_fill(m, 0.0)


def replacement_args(t, m):
    return (t, m)


def replacement_func():
    return dispatch_wrapper