"""
Pass: HaloLayoutFusion48
Full fusion of: conv2d(1x1) + pad + unfold(2,12,8) + unfold(3,12,8)
                + reshape(8,48,4,-1) + permute(0,2,3,1)
                + split([16,32],-1) + transpose(-1,-2)
Matches bfloat16 and float32 graphs: weight [384,256,1,1], input [1,256,16,16].

Uses shared Triton GEMM + layout-scatter kernels from halo_kernels.py.
Both pass files return the SAME halo_fused_dispatch function object to
avoid replacement_func_limit deduplication.
"""

import torch
from pass_dir.halo_kernels import halo_fused_dispatch  # shared dispatch


# -----------------------------------------------------------------------
# Pattern – must mirror bfloat16/float32 model.py exactly
# -----------------------------------------------------------------------
def pattern(in_0, in_1):
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = torch.nn.functional.pad(conv2d, [2, 2, 2, 2], 'constant', None)
    tmp_3 = tmp_2.unfold(2, 12, 8)
    tmp_4 = tmp_3.unfold(3, 12, 8)
    tmp_5 = tmp_4.reshape(8, 48, 4, -1)
    tmp_6 = tmp_5.permute(0, 2, 3, 1)
    split = torch.functional.split(tmp_6, [16, 32], dim=-1)
    tmp_8 = split[0]
    tmp_9 = split[1]
    tmp_10 = tmp_8.transpose(-1, -2)
    return (tmp_10, tmp_9)


def replacement_args(in_0, in_1):
    return (in_0, in_1, "r48")


def replacement_func():
    return halo_fused_dispatch