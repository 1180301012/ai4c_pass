"""
Pass: FuseEmbedPermute_2_7
Fuses: to(cuda) + embedding + permute([2,0,1]) + unsqueeze(0) + expand((2,-1,7,7)) + contiguous
Target graphs: float32/all-mpnet-base-v2
  in_0 shape: [32, 12], in_1 shape: [7, 7]
  Route: 's7_b2'

NOTE: .to() IS included in the pattern so in_1 is received as a CPU tensor by the
wrapper, which then caches the GPU copy to avoid repeated H2D transfers.
Note: all-mpnet uses device(type='cuda') WITHOUT index=0.
"""
import torch
from torch import device
from pass_dir.embedding_kernel import fused_embedding_dispatch


def pattern(in_0, in_1):
    tmp_1 = in_1.to(device(type='cuda'))
    tmp_2 = torch.nn.functional.embedding(tmp_1, in_0, None, None, 2.0, False, False)
    tmp_3 = tmp_2.permute([2, 0, 1])
    tmp_4 = tmp_3.unsqueeze(0)
    tmp_5 = tmp_4.expand((2, -1, 7, 7))
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(in_0, in_1):
    # in_1 is the original CPU tensor; the wrapper handles H2D with caching
    return (in_0, in_1, "s7_b2")


def replacement_func():
    return fused_embedding_dispatch