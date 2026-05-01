import torch
from pass_dir.silu_avgpool_flatten_impl import silu_avgpool_flatten


def pattern(in_0):
    tmp_0 = torch.nn.functional.silu(in_0, inplace=True)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, 1)
    tmp_2 = torch.flatten(tmp_1, 1)
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.5, False, True)
    return tmp_3


def replacement_args(in_0):
    return (in_0,)


def replacement_func():
    return silu_avgpool_flatten