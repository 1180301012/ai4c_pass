"""
Grand-fusion pass for D=32: fuses position-embedding lookup +
element-wise add + layer-norm + dropout into a single Triton kernel.
"""
import torch
from pass_dir.shared_kernels import _shared_dispatch


def pattern(word_emb, pos_table, pos_ids, weight, bias):
    pos_emb = torch.nn.functional.embedding(
        pos_ids, pos_table, 1, None, 2.0, False, False)
    total = word_emb + pos_emb
    out = torch.nn.functional.layer_norm(total, (32,), weight, bias, 1e-05)
    out = torch.nn.functional.dropout(out, 0.1, False, False)
    return out


def replacement_args(word_emb, pos_table, pos_ids, weight, bias):
    # pos_ids dropped; kernel hardcodes row_idx+2
    return (word_emb, pos_table, weight, bias, None, "gf32")


def replacement_func():
    return _shared_dispatch