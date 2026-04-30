import torch
import triton
import triton.language as tl
from pass_dir.shared_kernel import triton_dispatch


def pattern(causal_mask, tmp_15_input):
    tmp_16 = causal_mask[(slice(None, None, None), slice(None, None, None),
                           slice(None, None, None), slice(None, 21, None))]
    tmp_17 = tmp_16.masked_fill(tmp_15_input, -3.4028234663852886e+38)
    return tmp_17


def replacement_args(causal_mask, tmp_15_input):
    return (causal_mask, tmp_15_input, 21, "n21")


def replacement_func():
    return triton_dispatch