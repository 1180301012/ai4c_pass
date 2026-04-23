import torch
from pass_dir.shared_fused_bn_silu import dispatch_replacement


def pattern(in_0, in_1, in_2, in_3, x):
    tmp_5 = torch.nn.functional.batch_norm(x, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_6 = torch.nn.functional.silu(tmp_5, inplace=True)
    return (tmp_6,)


def replacement_args(in_0, in_1, in_2, in_3, x):
    return (in_0, in_1, in_2, in_3, x, "bn_silu")


def replacement_func():
    return dispatch_replacement