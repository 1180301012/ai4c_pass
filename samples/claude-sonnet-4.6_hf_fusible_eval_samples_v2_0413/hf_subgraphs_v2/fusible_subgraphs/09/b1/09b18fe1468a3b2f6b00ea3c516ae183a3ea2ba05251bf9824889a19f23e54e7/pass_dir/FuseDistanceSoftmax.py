"""
Pass: FuseDistanceSoftmax
Fuses: (in_1 - in_2).pow(2).sum(dim=3) * in_3  →  scaled distances [B,I,K]

The downstream softmax(dim=2) + unsqueeze(3) are cheap and stay as PyTorch
ops in the graph — no need to fuse them.  The win here is fusing the four
memory-bandwidth-heavy ops that create/consume large [1,4096,32,512] tensors.

in_1: [1, 4096, 32, 512]  float16/bfloat16
in_2: [1,    1, 32, 512]  float16/bfloat16
in_3: [1,    1, 32]       float16/bfloat16
out:  [1, 4096, 32]       float16/bfloat16   (= tmp_4 in model.py)
"""

import torch
from pass_dir.shared_kernels import shared_fused_kernel


def pattern(in_1, in_2, in_3):
    tmp_1 = in_1 - in_2
    tmp_2 = tmp_1.pow(2)
    tmp_3 = tmp_2.sum(dim=3)
    tmp_4 = in_3 * tmp_3
    return tmp_4


def replacement_args(in_1, in_2, in_3):
    # 6-arg wrapper: a=in_1, b=in_2, c=in_3, d,e=dummies, route="ds"
    return (in_1, in_2, in_3, in_1, in_1, "ds")


def replacement_func():
    return shared_fused_kernel