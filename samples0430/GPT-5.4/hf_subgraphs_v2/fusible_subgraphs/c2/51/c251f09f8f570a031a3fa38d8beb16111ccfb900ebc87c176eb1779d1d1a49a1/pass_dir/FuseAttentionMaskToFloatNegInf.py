import torch
from pass_dir.layoutlm_shared import shared_layoutlm_dispatch


def pattern(mask):
    tmp_12 = mask.to(dtype=torch.float32)
    tmp_13 = 1.0 - tmp_12
    tmp_14 = tmp_13 * -3.4028234663852886e+38
    return tmp_14


def replacement_args(mask):
    return (mask, "mask")


def replacement_func():
    return shared_layoutlm_dispatch