import torch
import triton
import triton.language as tl
from pass_dir.coat_shared_kernel import coat_fuse_A_dispatch, _coat_fuse_A_kernel


def pattern(in_2, in_3, conv_out, in_6):
    tmp_3 = torch.cat([in_2, in_3, conv_out], dim=1)
    tmp_4 = tmp_3.reshape(1, 8, 27, 49)
    tmp_5 = tmp_4.transpose(-1, -2)
    tmp_6 = in_6 * tmp_5
    return (tmp_6,)


def replacement_args(in_2, in_3, conv_out, in_6):
    return (in_2, in_3, conv_out, in_6, "A")


def replacement_func():
    return coat_fuse_A_dispatch