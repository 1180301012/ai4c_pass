import torch
import triton
import triton.language as tl
from pass_dir.ernie_shared_dispatch import ernie_dispatch


def pattern(in_0, in_1, in_2, in_3):
    tmp_16 = in_0 + in_1
    tmp_17 = torch.nn.functional.layer_norm(tmp_16, (32,), in_3, in_2, 1e-05)
    tmp_18 = torch.nn.functional.dropout(tmp_17, 0.1, False, False)
    return tmp_18


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3, "addln")


def replacement_func():
    return ernie_dispatch