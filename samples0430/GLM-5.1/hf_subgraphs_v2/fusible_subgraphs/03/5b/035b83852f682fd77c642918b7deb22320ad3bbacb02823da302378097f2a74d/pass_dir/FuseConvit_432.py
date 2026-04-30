import torch
from pass_dir.shared_kernels import fused_convit_dispatch


# ===== Pattern matching (layer_norm with normalized_shape=(432,)) =====

def pattern(in_0, in_1, in_2):
    return torch.nn.functional.layer_norm(in_2, (432,), in_1, in_0, 1e-06)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "route_432")


def replacement_func():
    return fused_convit_dispatch