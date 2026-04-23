import torch
from pass_dir.shared_fused_gelu_pad import replacement_func


def pattern(in_0):
    return torch.nn.functional.gelu(in_0)


def replacement_args(in_0):
    return (in_0, 'gelu_only_functional')