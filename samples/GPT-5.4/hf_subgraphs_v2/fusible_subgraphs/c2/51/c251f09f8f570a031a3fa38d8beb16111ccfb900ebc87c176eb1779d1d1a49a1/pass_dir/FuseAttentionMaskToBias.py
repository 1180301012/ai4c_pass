import torch
import triton
import triton.language as tl
from pass_dir.shared_layoutlm_runtime import shared_layoutlm_dispatch


# Match attention mask conversion exactly.
def pattern(attention_mask):
    tmp_12 = attention_mask.to(dtype=torch.float32)
    tmp_13 = 1.0 - tmp_12
    tmp_14 = tmp_13 * -3.4028234663852886e+38
    return tmp_14


def replacement_args(attention_mask):
    return (attention_mask, "mask")


def replacement_func():
    return shared_layoutlm_dispatch