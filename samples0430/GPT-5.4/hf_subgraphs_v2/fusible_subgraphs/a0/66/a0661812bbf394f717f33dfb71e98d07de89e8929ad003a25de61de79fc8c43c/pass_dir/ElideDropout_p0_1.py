import torch
from pass_dir.shared_dispatch import shared_replacement


def pattern(x):
    tmp = torch.nn.functional.dropout(x, 0.1, False, False)
    return tmp


def replacement_args(x):
    return (x, "dropout_identity")


def replacement_func():
    return shared_replacement