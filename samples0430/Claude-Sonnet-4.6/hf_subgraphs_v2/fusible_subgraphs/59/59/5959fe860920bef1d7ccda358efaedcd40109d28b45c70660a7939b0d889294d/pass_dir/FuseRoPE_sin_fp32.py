import torch
from pass_dir.shared_kernels import fused_dispatch


def pattern(catted):
    tmp_4 = catted.sin()
    tmp_5 = tmp_4 * 1.0
    tmp_7 = tmp_5.to(dtype=torch.float32)
    return tmp_7


def replacement_args(catted):
    return (catted, catted, "rope_sin_fp32")


def replacement_func():
    return fused_dispatch