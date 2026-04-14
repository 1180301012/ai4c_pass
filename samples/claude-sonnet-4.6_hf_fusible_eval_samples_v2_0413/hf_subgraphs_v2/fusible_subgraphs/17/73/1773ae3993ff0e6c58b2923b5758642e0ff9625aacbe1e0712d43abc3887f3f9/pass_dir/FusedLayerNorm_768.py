import torch
from pass_dir.shared_kernels import _dispatch


def pattern(in_0, in_1, in_4):
    # in_0 = bias [768], in_1 = weight [768], in_4 = input [..., 768]
    return torch.nn.functional.layer_norm(in_4, (768,), in_1, in_0, 1e-12)


def replacement_args(in_0, in_1, in_4):
    return (in_0, in_1, in_4, "ln_768")


def replacement_func():
    return _dispatch