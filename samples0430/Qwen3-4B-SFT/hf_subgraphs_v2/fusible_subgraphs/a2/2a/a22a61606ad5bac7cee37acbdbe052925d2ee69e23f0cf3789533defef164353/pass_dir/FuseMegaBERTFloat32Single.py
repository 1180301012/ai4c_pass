"""
Pass: FuseMegaBERTFloat32Single
Pattern: 3-embedding look-up  +  element-wise sum  +  dropout(training=F)  +  layer_norm(768,)  [float32]
Returns:  (ln_out,)              – float32 MegaBERT (weight=tmp_2, bias=tmp_1)
Routes:   "mega_768_single"
"""
import torch
from pass_dir.triton_emb_kernel import emb_fused_dispatch


def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    tmp_7  = torch.nn.functional.embedding(tmp_0, in_3, 0, None, 2.0, False, False)
    tmp_8  = torch.nn.functional.embedding(tmp_6, in_2, None, None, 2.0, False, False)
    tmp_9  = tmp_7 + tmp_8
    tmp_10 = torch.nn.functional.embedding(in_7, in_1, None, None, 2.0, False, False)
    tmp_9 += tmp_10
    tmp_12 = torch.nn.functional.dropout(tmp_9, 0.1, False, False)
    tmp_13 = torch.nn.functional.layer_norm(tmp_12, (768,), tmp_2, tmp_1, 1e-12)
    return (tmp_13,)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, "mega_768_single")


def replacement_func():
    return emb_fused_dispatch