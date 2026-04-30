import torch
import triton
import triton.language as tl
from pass_dir.shared_gemma_rotary import shared_dispatch


def pattern(in_5):
    tmp_10 = in_5[
        slice(None, None, None),
        slice(None, None, None),
        None,
        slice(None, None, None),
        slice(None, None, None),
    ]
    tmp_11 = tmp_10.expand(1, 1, 8, 3, 256)
    tmp_12 = tmp_11.reshape(1, 8, 3, 256)
    return tmp_12


def replacement_args(in_5):
    return (in_5, "broadcast_value")


def replacement_func():
    return shared_dispatch