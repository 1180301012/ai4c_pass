import torch
import sys
import os

# Add pass_dir to path for shared kernel import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _shared_kernel import fused_embedding_permute_expand_dispatch


def pattern(indices, weight):
    emb = torch.nn.functional.embedding(indices, weight, None, None, 2.0, False, False)
    perm = emb.permute([2, 0, 1])
    unsq = perm.unsqueeze(0)
    exp = unsq.expand((1, -1, 45, 45))
    cont = exp.contiguous()
    return cont


def replacement_args(indices, weight):
    return (indices, weight, "route_1_45_45")


def replacement_func():
    return fused_embedding_permute_expand_dispatch