import torch
from pass_dir._shared import shared_dispatch


def pattern(in_3, in_2, in_1):
    """Match: layer_norm(in_3, (768,), in_2, in_1, 1e-12) → single output tmp_4."""
    return torch.nn.functional.layer_norm(in_3, (768,), in_2, in_1, 1e-12)


def replacement_args(in_3, in_2, in_1):
    return (in_3, in_2, in_1, "layer_norm")


def replacement_func():
    return shared_dispatch