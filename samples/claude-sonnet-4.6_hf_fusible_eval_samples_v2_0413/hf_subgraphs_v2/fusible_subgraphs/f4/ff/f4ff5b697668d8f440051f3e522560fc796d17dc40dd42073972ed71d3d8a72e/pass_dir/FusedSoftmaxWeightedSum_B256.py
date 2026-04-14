import torch
import triton
import triton.language as tl
from pass_dir.kernel_impl import dispatch_fused


def pattern(in_0, in_1, tmp_3):
    tmp_4 = tmp_3.mul(in_0)
    tmp_5 = tmp_4.reshape(256, 17, -1)
    tmp_6 = torch.sum(tmp_5, dim=2, keepdim=True)
    tmp_7 = tmp_3.mul(in_1)
    tmp_8 = tmp_7.reshape(256, 17, -1)
    tmp_9 = torch.sum(tmp_8, dim=2, keepdim=True)
    tmp_10 = torch.cat([tmp_6, tmp_9], dim=-1)
    return tmp_10


def replacement_args(in_0, in_1, tmp_3):
    return (in_0, in_1, tmp_3, "weighted")


def replacement_func():
    return dispatch_fused