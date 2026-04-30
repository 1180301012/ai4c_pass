import torch
import triton
import triton.language as tl

from pass_dir.shared_cached_full_graph_leaf import cached_full_graph_wrapper


def pattern(in_0, in_1, in_2):
    tmp_0 = in_1.view(-1, 1)
    tmp_1 = tmp_0 * in_2
    tmp_2 = in_0.view((-1, 1))
    tmp_3 = tmp_2.expand_as(tmp_1)
    tmp_4 = tmp_1.new_zeros((1000, 16))
    return (tmp_3, tmp_4, tmp_1)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, 1000, 16)


def replacement_func():
    return cached_full_graph_wrapper