import torch
from pass_dir.kernels import dispatch_ln


def pattern(bias, weight, x):
    # x is tmp_8 (the residual sum, computed externally by the add)
    tmp_9 = torch.nn.functional.layer_norm(x, (384,), weight, bias, 1e-05)
    return tmp_9


def replacement_args(bias, weight, x):
    return (x, weight, bias, 1024)


def replacement_func():
    return dispatch_ln