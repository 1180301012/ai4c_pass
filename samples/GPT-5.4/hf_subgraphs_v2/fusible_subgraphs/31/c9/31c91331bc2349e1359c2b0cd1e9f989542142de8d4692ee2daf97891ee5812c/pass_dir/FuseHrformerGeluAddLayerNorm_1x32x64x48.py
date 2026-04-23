import torch
import triton
import triton.language as tl

from pass_dir.shared_hrformer_prefix import hrformer_prefix_route


def pattern(in_2, in_3):
    tmp_2 = torch.nn.functional.gelu(in_2, approximate='none')
    tmp_3 = tmp_2.flatten(2)
    tmp_4 = tmp_3.transpose(1, 2)
    tmp_5 = tmp_4.contiguous()
    tmp_6 = in_3 + tmp_5
    return tmp_6


def replacement_args(in_2, in_3):
    return (in_2, in_3)


def replacement_func():
    return hrformer_prefix_route