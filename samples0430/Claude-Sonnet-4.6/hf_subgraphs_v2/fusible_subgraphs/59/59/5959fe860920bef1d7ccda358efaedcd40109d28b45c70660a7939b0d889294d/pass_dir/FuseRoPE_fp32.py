import torch
from pass_dir.shared_kernels import fused_dispatch


def pattern(catted):
    tmp_2 = catted.cos()
    tmp_3 = tmp_2 * 1.0
    tmp_6 = tmp_3.to(dtype=torch.float32)
    return tmp_6


def replacement_args(catted):
    return (catted, catted, "rope_cos_fp32")


def replacement_func():
    return fused_dispatch