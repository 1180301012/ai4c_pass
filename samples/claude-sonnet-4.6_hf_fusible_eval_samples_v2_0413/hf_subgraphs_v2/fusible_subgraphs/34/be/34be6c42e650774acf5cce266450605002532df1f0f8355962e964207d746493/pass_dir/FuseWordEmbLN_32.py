"""
Mega-fusion pass for D=32:
  word-embedding lookup + pos-embedding lookup + add + layer-norm + dropout
  => single Triton kernel.
"""
import torch
from pass_dir.shared_kernels import _shared_dispatch


def pattern(in_0, in_4, pos_table, pos_ids, weight, bias):
    word_emb = torch.nn.functional.embedding(
        in_0, in_4, 1, None, 2.0, False, False)
    pos_emb = torch.nn.functional.embedding(
        pos_ids, pos_table, 1, None, 2.0, False, False)
    total = word_emb + pos_emb
    out = torch.nn.functional.layer_norm(total, (32,), weight, bias, 1e-05)
    out = torch.nn.functional.dropout(out, 0.1, False, False)
    return out


def replacement_args(in_0, in_4, pos_table, pos_ids, weight, bias):
    # dispatch(a=in_0, b=in_4, c=pos_table, d=weight, e=bias, route)
    return (in_0, in_4, pos_table, weight, bias, "wgf32")


def replacement_func():
    return _shared_dispatch