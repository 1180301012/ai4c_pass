import torch
from pass_dir.shared_dispatch import shared_replacement_dispatch


# Match the activation-producing subgraph; tmp_10 is observable externally.
def pattern(x, running_mean, running_var, bias, weight):
    tmp_10 = torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    tmp_10 = torch.nn.functional.silu(tmp_10, inplace=True)
    return tmp_10


def replacement_args(x, running_mean, running_var, bias, weight):
    return (x, running_mean, running_var, bias, weight, "bn_silu")


def replacement_func():
    return shared_replacement_dispatch