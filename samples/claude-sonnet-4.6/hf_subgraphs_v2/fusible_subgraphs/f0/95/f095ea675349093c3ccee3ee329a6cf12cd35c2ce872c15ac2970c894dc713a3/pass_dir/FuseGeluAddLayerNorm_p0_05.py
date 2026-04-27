import torch
import triton
import triton.language as tl
from pass_dir.shared_dispatch import fused_dispatch


def pattern(tmp_4, in_3):
    tmp_5 = torch.nn.functional.gelu(tmp_4)
    tmp_6 = tmp_5.transpose(1, 2)
    tmp_7 = in_3 + tmp_6
    tmp_8 = torch.nn.functional.dropout(tmp_7, 0.05, False, False)
    return tmp_8


def replacement_args(tmp_4, in_3):
    # a3 = in_3 is a dummy to match the shared 4-arg dispatch signature
    return (tmp_4, in_3, in_3, "gelu_add")


def replacement_func():
    return fused_dispatch