"""
Fused pass for BigBird style embedding + layer_norm subgraph.
  - 3 embedding lookups (word, token-type, position)
  - elementwise sum
  - dropout (training=False => identity)
  - layer_norm
Returns ONLY layer_norm_out (1 output), hidden_size=768.

Matches: float16, bfloat16, float32 BigBird graphs (hidden_size=768).
Pattern takes ALL 8 model inputs as explicit params so torch.fx traces them.
Uses shared dispatch_fused_emb_ln (route="768_1out") to avoid replacement_func_limit.
"""
import torch
import triton
import triton.language as tl
from pass_dir.FusedEmbeddingLayerNorm_kernel import dispatch_fused_emb_ln


# ---------------------------------------------------------------------------
# Pattern – mirrors BigBird float16 model.py exactly.
# ALL 8 inputs are explicit params.
# BigBird model arg order: (in_0, in_5, in_4, in_3, in_2, in_1, in_6, in_7)
#   in_0: word indices, in_5: word emb, in_4: tt emb, in_3: pos emb,
#   in_2: ln weight, in_1: ln bias, in_6: tt indices, in_7: pos indices
# ---------------------------------------------------------------------------
def pattern(in_0, in_5, in_4, in_3, in_2, in_1, in_6, in_7):
    tmp_7  = torch.nn.functional.embedding(in_0, in_5, 0,    None, 2.0, False, False)
    tmp_8  = torch.nn.functional.embedding(in_6, in_4, None, None, 2.0, False, False)
    tmp_9  = tmp_7 + tmp_8
    tmp_10 = torch.nn.functional.embedding(in_7, in_3, None, None, 2.0, False, False)
    tmp_9 += tmp_10
    tmp_12 = torch.nn.functional.dropout(tmp_9, 0.1, False, False)
    tmp_13 = torch.nn.functional.layer_norm(tmp_12, (768,), in_2, in_1, 1e-12)
    return tmp_13


# ---------------------------------------------------------------------------
# replacement_args – BigBird order -> canonical order + route string
# BigBird: (in_0, in_5, in_4, in_3, in_2, in_1, in_6, in_7)
#         => (word_idx, word_emb, tt_emb, pos_emb, ln_weight, ln_bias, ...)
# ---------------------------------------------------------------------------
def replacement_args(in_0, in_5, in_4, in_3, in_2, in_1, in_6, in_7):
    # (word_idx, word_emb, tt_emb, pos_emb, ln_weight, ln_bias, route)
    return (in_0, in_5, in_4, in_3, in_2, in_1, "768_1out")


# ---------------------------------------------------------------------------
# replacement_func – returns the shared dispatch wrapper
# ---------------------------------------------------------------------------
def replacement_func():
    return dispatch_fused_emb_ln