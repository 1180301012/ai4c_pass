import torch
import triton
import triton.language as tl

from pass_dir.shared_ln_tail import replacement_dispatch


def pattern(conv_out, in_4, in_1, in_0):
    tmp_5 = conv_out + in_4
    tmp_6 = tmp_5.flatten(2)
    tmp_7 = tmp_6.transpose(1, 2)
    tmp_8 = torch.nn.functional.layer_norm(tmp_7, (1024,), in_1, in_0, 1e-05)
    tmp_9 = tmp_8.transpose(0, 1)
    tmp_10 = tmp_8.transpose(0, 1)
    return (tmp_7, tmp_10, tmp_9)


def replacement_args(conv_out, in_4, in_1, in_0):
    return (conv_out, in_4, in_1, in_0)


def replacement_func():
    return replacement_dispatch