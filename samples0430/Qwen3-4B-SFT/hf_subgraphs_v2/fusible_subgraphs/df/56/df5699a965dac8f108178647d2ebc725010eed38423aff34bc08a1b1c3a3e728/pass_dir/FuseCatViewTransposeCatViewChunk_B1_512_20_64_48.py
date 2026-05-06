"""
Fuse:  cat([in_2, in_4], dim=1)
       -> view(1, 2, 20, 64, 48)
       -> transpose(1, 2)
       -> contiguous()
       -> view(1, 40, 64, 48)
       -> chunk(2, dim=1)

into a single Triton channel-shuffle kernel.
Returns (out0, out1) each [1, 20, 64, 48].
"""

import torch
from pass_dir.shuffle_kernel import shuffler_C20_HW3072


def pattern(in_2, in_4):
    tmp_5  = torch.cat([in_2, in_4], dim=1)
    tmp_7  = tmp_5.view(1, 2, 20, 64, 48)
    tmp_8  = torch.transpose(tmp_7, 1, 2)
    tmp_9  = tmp_8.contiguous()
    tmp_10 = tmp_9.view(1, 40, 64, 48)
    chunk  = tmp_10.chunk(2, dim=1)
    tmp_16 = chunk[0]
    tmp_17 = chunk[1]
    return tmp_16, tmp_17


def replacement_args(in_2, in_4):
    return (in_2, in_4)


@torch.fx.wrap
def fused_cat_shuffle_chunk_b1_512_20_64_48(in_2, in_4):
    out0, out1 = shuffler_C20_HW3072(in_2, in_4)
    return out0, out1


def replacement_func():
    return fused_cat_shuffle_chunk_b1_512_20_64_48