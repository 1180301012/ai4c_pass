"""
Pass: FuseEmbedPermute_1_11
Fuses: to(cuda:0) + embedding + permute([2,0,1]) + unsqueeze(0) + expand((1,-1,11,11)) + contiguous
Target graphs: float16/mpnet-base and bfloat16/mpnet-base
  in_0 shape: [32, 12], in_1 shape: [11, 11]
  Route: 's11_b1'

NOTE: .to() IS included in the pattern so in_1 is received as a CPU tensor by the
wrapper, which then caches the GPU copy to avoid repeated H2D transfers.
"""
import torch
from torch import device
from pass_dir.embedding_kernel import fused_embedding_dispatch


def pattern(in_0, in_1):
    tmp_1 = in_1.to(device(type='cuda', index=0))
    tmp_2 = torch.nn.functional.embedding(tmp_1, in_0, None, None, 2.0, False, False)
    tmp_3 = tmp_2.permute([2, 0, 1])
    tmp_4 = tmp_3.unsqueeze(0)
    tmp_5 = tmp_4.expand((1, -1, 11, 11))
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(in_0, in_1):
    # in_1 is the original CPU tensor; the wrapper handles H2D with caching
    return (in_0, in_1, "s11_b1")


def replacement_func():
    return fused_embedding_dispatch