import torch
import operator
from pass_dir.shared_kernels import _shared_dispatch


def pattern(x):
    a = x.__eq__(1)
    b = a.to(torch.float32)
    b = operator.imul(b, -3.4028234663852886e+38)
    c = b.unsqueeze(1)
    d = c.unsqueeze(1)
    return d


def replacement_args(x):
    return (x, None, None, None, "mask")


def replacement_func():
    return _shared_dispatch