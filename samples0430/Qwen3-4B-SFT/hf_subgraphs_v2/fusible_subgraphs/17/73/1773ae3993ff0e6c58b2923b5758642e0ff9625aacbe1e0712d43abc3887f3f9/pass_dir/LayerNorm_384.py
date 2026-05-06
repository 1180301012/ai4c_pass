import torch
from pass_dir._shared_ln import dispatch_ln


def pattern(in_0, in_1, in_4):
    tmp_3 = torch.nn.functional.layer_norm(in_4, (384,), in_1, in_0, 1e-12)
    return tmp_3


def replacement_args(in_0, in_1, in_4):
    return (in_0, in_1, in_4, "D384")


def replacement_func():
    return dispatch_ln