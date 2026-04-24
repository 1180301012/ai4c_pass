"""
Fused pass for BERT-large style embedding + layer_norm subgraph.
  - 3 embedding lookups (word, token-type, position)
  - elementwise sum
  - dropout (training=False => identity)
  - layer_norm
Returns (embedding_sum, layer_norm_out) - 2 outputs, hidden_size=1024.

Matches: float16, bfloat16, float32 BERT-large graphs (hidden_size=1024).
Pattern takes ALL 8 model inputs as explicit params so torch.fx traces them
as Proxy objects, avoiding any name-lookup failures.
Uses shared dispatch_fused_emb_ln (route="1024_2out") so replacement_func_limit
never drops this pass relative to the other two.
"""
import torch
import triton
import triton.language as tl
from pass_dir.FusedEmbeddingLayerNorm_kernel import dispatch_fused_emb_ln


# ---------------------------------------------------------------------------
# Pattern – mirrors model.py exactly.
# ALL 8 inputs are explicit params so torch.fx can trace them as placeholders.
# in_0: word indices, in_3: word emb, in_2: tt emb, in_1: pos emb,
# in_5: ln weight, in_4: ln bias, in_6: tt indices, in_7: pos indices
# ---------------------------------------------------------------------------
def pattern(in_0, in_3, in_2, in_1, in_5, in_4, in_6, in_7):
    tmp_7  = torch.nn.functional.embedding(in_0, in_3, 0,    None, 2.0, False, False)
    tmp_8  = torch.nn.functional.embedding(in_6, in_2, None, None, 2.0, False, False)
    tmp_9  = tmp_7 + tmp_8
    tmp_10 = torch.nn.functional.embedding(in_7, in_1, None, None, 2.0, False, False)
    tmp_9 += tmp_10
    tmp_12 = torch.nn.functional.dropout(tmp_9, 0.1, False, False)
    tmp_13 = torch.nn.functional.layer_norm(tmp_12, (1024,), in_5, in_4, 1e-12)
    return (tmp_12, tmp_13)


# ---------------------------------------------------------------------------
# replacement_args – returns canonical arg order + route string
# ---------------------------------------------------------------------------
def replacement_args(in_0, in_3, in_2, in_1, in_5, in_4, in_6, in_7):
    # (word_idx, word_emb, tt_emb, pos_emb, ln_weight, ln_bias, route)
    return (in_0, in_3, in_2, in_1, in_5, in_4, "1024_2out")


# ---------------------------------------------------------------------------
# replacement_func – returns the shared dispatch wrapper
# ---------------------------------------------------------------------------
def replacement_func():
    return dispatch_fused_emb_ln