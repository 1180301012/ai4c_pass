"""
Pass: FuseTinyBERT64Double
Pattern: 3-embedding look-up  +  element-wise sum  +  dropout(training=F)  +  layer_norm(64,)
Returns:  (emb_sum, ln_out)              – Tiny Megatron 64D variants (float16 & float32)
Routes:   "tiny_64_double"
"""
import torch
from pass_dir.triton_emb_kernel import emb_fused_dispatch


def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    tmp_7  = torch.nn.functional.embedding(in_0, in_3, 0,  None, 2.0, False, False)
    tmp_8  = torch.nn.functional.embedding(in_6, in_2, None, None, 2.0, False, False)
    tmp_9  = tmp_7 + tmp_8
    tmp_10 = torch.nn.functional.embedding(in_7, in_1, None, None, 2.0, False, False)
    tmp_9 += tmp_10
    tmp_12 = torch.nn.functional.dropout(tmp_9, 0.1, False, False)
    tmp_13 = torch.nn.functional.layer_norm(tmp_12, (64,), in_5, in_4, 1e-12)
    return (tmp_12, tmp_13)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, "tiny_64_double")


def replacement_func():
    return emb_fused_dispatch