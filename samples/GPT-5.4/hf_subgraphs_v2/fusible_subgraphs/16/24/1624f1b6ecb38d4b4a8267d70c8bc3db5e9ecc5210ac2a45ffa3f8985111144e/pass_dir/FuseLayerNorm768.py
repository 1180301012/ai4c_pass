import torch

from pass_dir.vivit_shared import shared_replacement_func


def pattern(x, ln_bias, ln_weight):
    return torch.nn.functional.layer_norm(x, (768,), ln_weight, ln_bias, 1e-06)


def replacement_args(x, ln_bias, ln_weight):
    return (x, ln_weight, ln_bias, "layer_norm_768")


def replacement_func():
    return shared_replacement_func()