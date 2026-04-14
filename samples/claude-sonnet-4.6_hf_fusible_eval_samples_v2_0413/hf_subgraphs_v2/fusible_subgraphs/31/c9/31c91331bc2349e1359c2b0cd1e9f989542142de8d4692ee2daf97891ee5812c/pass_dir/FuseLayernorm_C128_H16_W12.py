import torch
from pass_dir._shared_kernels import dispatch_kernel, _dt


# LayerNorm + view pass for C=128. Returns a SINGLE tensor (tmp_12).

def pattern(in_0, in_1, x):
    tmp_11 = torch.nn.functional.layer_norm(x, (128,), in_1, in_0, 1e-06)
    return tmp_11.view(1, 16, 12, 128)


def replacement_args(in_0, in_1, x):
    return (in_0, in_1, x, "ln_128_{}".format(_dt(x)))


def replacement_func():
    return dispatch_kernel