import torch
import triton
import triton.language as tl
from pass_dir.shared_unfold_kernel import triton_unfold_dispatch


def pattern(x):
    tmp_0 = x.contiguous()
    tmp_1 = tmp_0.unsqueeze(-1)
    tmp_2 = torch.nn.functional.unfold(tmp_1, kernel_size=[9, 1], dilation=1, padding=[4, 0], stride=1)
    tmp_3 = tmp_2.transpose(1, 2)
    tmp_4 = tmp_3.reshape(1, -1, 16, 9)
    tmp_5 = torch.reshape(tmp_4, [-1, 8, 9])
    return tmp_5


def replacement_args(x):
    return (x, '16_45')


def replacement_func():
    return triton_unfold_dispatch