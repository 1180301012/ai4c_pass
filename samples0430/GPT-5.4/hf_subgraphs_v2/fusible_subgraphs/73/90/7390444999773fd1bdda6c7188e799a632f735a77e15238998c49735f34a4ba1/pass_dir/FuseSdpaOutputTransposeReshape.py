import torch
import triton
import triton.language as tl
from pass_dir.shared_sdpa_layout import replacement_func


def pattern(x):
    t = x.transpose(1, 2)
    y = t.reshape(x.shape[0], x.shape[2], x.shape[1] * x.shape[3])
    return y


def replacement_args(x):
    return (x, 'sdpa_unpack_last')