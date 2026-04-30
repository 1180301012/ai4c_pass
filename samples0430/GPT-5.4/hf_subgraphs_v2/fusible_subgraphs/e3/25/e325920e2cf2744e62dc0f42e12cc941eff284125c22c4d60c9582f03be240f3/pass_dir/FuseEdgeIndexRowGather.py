import torch

from pass_dir.edge_index_row_gather_impl import edge_index_row_gather


def pattern(in_0, in_1):
    tmp_1 = in_0[0]
    tmp_2 = in_1.index_select(-2, tmp_1)
    return tmp_2


def replacement_args(in_0, in_1):
    return (in_0[0], in_1)


def replacement_func():
    return edge_index_row_gather