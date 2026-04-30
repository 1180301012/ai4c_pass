import torch
import triton
import triton.language as tl

from pass_dir.qkv_postprocess_shared import qkv_postprocess_dispatch


def pattern(in_0, in_1):
    linear = torch.nn.functional.linear(in_1, in_0, None)
    tmp_2 = linear.reshape(1, 197, 3, 16, 48)
    tmp_3 = tmp_2.permute(2, 0, 3, 1, 4)
    unbind = tmp_3.unbind(0)
    tmp_5 = unbind[0]
    tmp_6 = unbind[1]
    tmp_7 = unbind[2]
    tmp_8 = tmp_6.transpose(-2, -1)
    return (tmp_5, tmp_8, tmp_7)


def replacement_args(in_0, in_1):
    return (in_0, in_1, 16)


def replacement_func():
    return qkv_postprocess_dispatch