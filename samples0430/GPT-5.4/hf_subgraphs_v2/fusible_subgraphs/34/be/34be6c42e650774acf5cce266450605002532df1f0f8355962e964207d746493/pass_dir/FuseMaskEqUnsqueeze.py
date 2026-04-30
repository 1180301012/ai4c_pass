import torch
import triton
import triton.language as tl
from pass_dir.ernie_shared_dispatch import ernie_dispatch


def pattern(in_0):
    tmp_5 = in_0.__eq__(1)
    tmp_6 = tmp_5.to(torch.float32)
    tmp_6 = torch.mul(tmp_6, -3.4028234663852886e+38)
    tmp_7 = tmp_6
    tmp_8 = tmp_7.unsqueeze(1)
    tmp_9 = tmp_8.unsqueeze(1)
    return tmp_9


def replacement_args(in_0):
    return (in_0, "mask")


def replacement_func():
    return ernie_dispatch