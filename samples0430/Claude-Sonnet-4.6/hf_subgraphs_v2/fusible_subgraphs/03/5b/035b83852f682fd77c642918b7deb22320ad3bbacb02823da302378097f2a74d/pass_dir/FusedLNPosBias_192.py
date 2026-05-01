import torch
from pass_dir.shared_ln_kernel import ln_triton_dispatch  # shared dispatch


def pattern(in_0, in_1, in_2):
    return torch.nn.functional.layer_norm(in_2, (192,), in_1, in_0, 1e-06)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "route_192")


def replacement_func():
    return ln_triton_dispatch