import torch
import triton
import triton.language as tl
from pass_dir.kernels import triton_attention_mask


def pattern(in_5):
    tmp_4 = in_5.to(torch.float32)
    tmp_5 = torch.tensor(1.0, dtype=torch.float32)
    tmp_6 = tmp_5 - tmp_4
    tmp_7 = tmp_6.to(torch.bool)
    tmp_8 = tmp_6.masked_fill(tmp_7, -3.4028234663852886e+38)
    return tmp_8


def replacement_args(in_5):
    return (in_5,)


def replacement_func():
    return triton_attention_mask