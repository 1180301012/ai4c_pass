import torch
from pass_dir.shared_dispatch import shared_replacement_dispatch


# Pattern includes both observable outputs of the matched subgraph
def pattern(x):
    tmp_10 = torch.nn.functional.silu(x, inplace=True)
    tmp_11 = tmp_10.mean((2, 3), keepdim=True)
    return (tmp_10, tmp_11)


def replacement_args(x):
    return (x, "silu_mean")


def replacement_func():
    return shared_replacement_dispatch