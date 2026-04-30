import torch
from pass_dir.shared_kernels import fused_embedding_expand_dispatch


def pattern(weight, indices):
    emb = torch.nn.functional.embedding(indices, weight, None, None, 2.0, False, False)
    p = emb.permute([2, 0, 1])
    u = p.unsqueeze(0)
    e = u.expand((1, -1, 11, 11))
    c = e.contiguous()
    return c


def replacement_args(weight, indices):
    return (weight, indices, "r_1_11_11")


def replacement_func():
    return fused_embedding_expand_dispatch