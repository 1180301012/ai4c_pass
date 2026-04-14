import torch
from pass_dir.shared_kernels import _shared_dispatch


def pattern(x, y, weight, bias):
    tmp = x + y
    out = torch.nn.functional.layer_norm(tmp, (32,), weight, bias, 1e-05)
    out = torch.nn.functional.dropout(out, 0.1, False, False)
    return out


def replacement_args(x, y, weight, bias):
    return (x, y, weight, bias, None, "aln32")


def replacement_func():
    return _shared_dispatch