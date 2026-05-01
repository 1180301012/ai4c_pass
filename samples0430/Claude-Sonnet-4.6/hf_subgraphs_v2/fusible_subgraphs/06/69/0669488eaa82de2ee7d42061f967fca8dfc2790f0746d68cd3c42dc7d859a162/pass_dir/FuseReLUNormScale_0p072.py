import torch
from pass_dir.shared_dispatch import relu_norm_scale_dispatch


def pattern(in_0, in_1, in_2):
    # in_1 = tmp_2 (flattened relu output [B,N,D]), in_2 = tmp_3 (norm [B,N,1])
    tmp_4 = in_2 * 0.07216878364870322
    tmp_5 = tmp_4.clamp(min=1e-05)
    tmp_6 = in_1 / tmp_5
    tmp_7 = tmp_6 * in_0
    return tmp_7


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "s072")


def replacement_func():
    return relu_norm_scale_dispatch