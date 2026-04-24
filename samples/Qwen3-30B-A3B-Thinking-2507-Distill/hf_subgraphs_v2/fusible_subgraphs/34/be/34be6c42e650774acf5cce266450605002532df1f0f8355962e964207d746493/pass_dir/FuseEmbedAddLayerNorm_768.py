import torch
import triton
import triton.language as tl
from pass_dir.embed_add_ln_impl import fused_embed_add_ln


def pattern(in_0, in_2, in_1, in_3, in_4, pos_indices):
    """
    pos_indices: [1, 15] int64 – pre-computed position indices (2..16)
    """
    tmp_10 = torch.nn.functional.embedding(in_0, in_4, 1, None, 2.0, False, False)
    tmp_15 = torch.nn.functional.embedding(pos_indices, in_3, 1, None, 2.0, False, False)
    tmp_16 = tmp_10 + tmp_15
    tmp_17 = torch.nn.functional.layer_norm(tmp_16, (768,), in_2, in_1, 1e-05)
    tmp_18 = torch.nn.functional.dropout(tmp_17, 0.1, False, False)
    return tmp_18


def replacement_args(in_0, in_2, in_1, in_3, in_4, pos_indices):
    return (in_0, in_2, in_1, in_3, in_4, pos_indices, "h768")


def replacement_func():
    return fused_embed_add_ln


# ── Keep trailing blank line to avoid unexpected edits ───────────────────────