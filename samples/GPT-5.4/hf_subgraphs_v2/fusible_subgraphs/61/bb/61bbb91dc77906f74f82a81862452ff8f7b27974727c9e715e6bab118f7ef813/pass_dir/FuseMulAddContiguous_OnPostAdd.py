import torch
from pass_dir.shared_fused_einsum_epilogue_v2 import replacement_func


def pattern(in_5, in_0, in_2):
    tmp_3 = in_5 * in_0
    tmp_4 = tmp_3 + in_2
    tmp_5 = tmp_4.contiguous()
    return tmp_5


def replacement_args(in_5, in_0, in_2):
    return (in_5, in_0, in_2, "route_pointwise_tensor")