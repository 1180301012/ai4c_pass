import torch
from pass_dir.shared_sdpa_layout import replacement_func


def pattern(x):
    t = x.transpose(1, 2)
    y = t.reshape(16, 128, 128)
    return y


def replacement_args(x):
    return (x, 'sdpa_unpack_last')