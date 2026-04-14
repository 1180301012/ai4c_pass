"""
Pass: FuseEmbedPermute_1_45
Fuses: to(cuda:0) + embedding + permute([2,0,1]) + unsqueeze(0) + expand((1,-1,45,45)) + contiguous
Target graphs: float16/tiny-random and bfloat16/tiny-random MPNetForSequenceClassification
  in_0 shape: [32, 4], in_1 shape: [45, 45]
  Route: 's45_b1'

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
    tmp_5 = tmp_4.expand((1, -1, 45, 45))
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(in_0, in_1):
    # in_1 is the original CPU tensor; the wrapper handles H2D with caching
    return (in_0, in_1, "s45_b1")


def replacement_func():
    return fused_embedding_dispatch