"""
Pass: FuseEmbeddingContiguous_1_45

Fuses embedding + permute([2,0,1]) + unsqueeze(0) + expand((1,-1,45,45)) + contiguous
into a single Triton kernel via shared dispatch.

Target models: float16 and bfloat16 tiny-random-MPNetForSequenceClassification
  in_0: [32, 4]  (weight)
  in_1: [45, 45] (indices, int64, already on cuda after .to())
  output: [1, 4, 45, 45]
"""
import torch
from pass_dir.emb_kernel import fused_emb_dispatch


def pattern(indices, weight):
    tmp_2 = torch.nn.functional.embedding(indices, weight, None, None, 2.0, False, False)
    tmp_3 = tmp_2.permute([2, 0, 1])
    tmp_4 = tmp_3.unsqueeze(0)
    tmp_5 = tmp_4.expand((1, -1, 45, 45))
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(indices, weight):
    return (indices, weight, "1_45")


def replacement_func():
    return fused_emb_dispatch