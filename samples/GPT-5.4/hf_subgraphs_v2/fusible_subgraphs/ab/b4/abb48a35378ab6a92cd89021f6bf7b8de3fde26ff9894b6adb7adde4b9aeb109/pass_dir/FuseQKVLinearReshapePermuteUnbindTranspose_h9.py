import torch
import triton
import triton.language as tl

from pass_dir.qkv_linear_layout_shared import replacement_func


def pattern(in_0, in_1):
    linear = torch.nn.functional.linear(in_1, in_0, None)
    tmp_2 = linear.reshape(1, 197, 3, 9, 48)
    tmp_3 = tmp_2.permute(2, 0, 3, 1, 4)
    unbind = tmp_3.unbind(0)
    return unbind


def replacement_args(in_0, in_1):
    return (in_0, in_1, "h9")