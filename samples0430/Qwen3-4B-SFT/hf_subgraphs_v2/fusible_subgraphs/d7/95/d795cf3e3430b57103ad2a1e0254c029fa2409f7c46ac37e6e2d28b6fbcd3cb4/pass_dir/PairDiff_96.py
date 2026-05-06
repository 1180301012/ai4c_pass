"""
Pass: PairDiff_96
Matches the pair_diff pattern for the 96-channel variant (used in fp16/bf16 graphs).

Pattern matched on the 19×19=361, 7×7=49 post-reshape shape.
The pattern and the float32 variant (C=128) do NOT share any code here —
they are separate files because the reshape constant 96 is embedded in the graph.
"""
import torch
import triton
import triton.language as tl
from pass_dir.pair_diff_kernel import pair_diff_wrapper, dispatch_wrapper


def pattern(tmp_9):
    tmp_10 = tmp_9.unsqueeze(2)
    tmp_11 = tmp_9.unsqueeze(3)
    tmp_12 = tmp_10 - tmp_11
    tmp_13 = tmp_12 != 0
    tmp_14 = tmp_12.masked_fill(tmp_13, -1000.0)
    return tmp_14


def replacement_args(tmp_9):
    return (tmp_9, None)


def replacement_func():
    return dispatch_wrapper