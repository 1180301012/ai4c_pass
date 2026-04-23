import torch
from pass_dir.shared_fused_einsum_epilogue_v2 import replacement_func


def pattern(in_4, in_1):
    einsum = torch.functional.einsum('bchj,bhwj->bchw', in_4, in_1)
    return einsum


def replacement_args(in_4, in_1):
    return (in_4, in_1, "route_einsum")