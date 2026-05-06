import torch
import triton
import triton.language as tl
from pass_dir.shared_kernel import _shared_dispatch


def pattern(in_0, in_1):
    tmp_1 = in_1.to(torch.device('cuda', 0))
    tmp_2 = torch.nn.functional.embedding(tmp_1, in_0, None, None, 2.0, False, False)
    tmp_3 = tmp_2.permute([2, 0, 1])
    tmp_4 = tmp_3.unsqueeze(0)
    tmp_5 = tmp_4.expand((1, -1, 45, 45))
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(in_0, in_1):
    return (in_0, in_1, "e4_s45")


def replacement_func():
    return _shared_dispatch