import torch
import triton
import triton.language as tl

from pass_dir.shared_cached_zero import cached_new_zeros


def pattern(tmp_1):
    tmp_4 = tmp_1.new_zeros((128, 128))
    return tmp_4


def replacement_args(tmp_1):
    return (tmp_1, 128, 128)


def replacement_func():
    return cached_new_zeros