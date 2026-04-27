import torch
from pass_dir.shared_kernels import pass_dispatch


def pattern(in_0, in_2, in_3, emb):
    """Match: token_emb + pos_emb  →  layer_norm"""
    tmp_13 = in_0 + emb
    tmp_14 = torch.nn.functional.layer_norm(tmp_13, (16,), in_3, in_2, 1e-05)
    return tmp_14


def replacement_args(in_0, in_2, in_3, emb):
    # pass_dispatch(in_0, emb, weight=in_3, bias=in_2, route)
    return (in_0, emb, in_3, in_2, "add_ln_16")


def replacement_func():
    return pass_dispatch