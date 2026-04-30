import torch
from pass_dir.shared_fused_kernel import fused_gelu_mean_dispatch


def pattern(in_0):
    tmp_0 = torch.nn.functional.gelu(in_0)
    return tmp_0


def replacement_args(in_0):
    return (in_0, "gelu_with_cache")


def replacement_func():
    return fused_gelu_mean_dispatch