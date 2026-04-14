"""
Fused pass: 3×embedding + add + dropout(training=False) + layer_norm(768)
Returns ONE tensor: (ln_out,)

Matches: Mahmoud8_bigbird-roberta-base (float16/bfloat16/float32)
  in_0 = word_ids, in_5 = word_emb_weight (padding_idx=0)
  in_6 = tok_ids,  in_4 = tok_type_emb_weight
  in_7 = pos_ids,  in_3 = pos_emb_weight
  in_2 = ln_weight, in_1 = ln_bias
"""
import torch
from pass_dir.embed_add_ln_kernel import run_fused_embed_add_ln

_D = 768
_BLOCK_D = 1024  # next power-of-2 >= 768


# ── Pattern ──────────────────────────────────────────────────────────────────
def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    import torch.nn.functional as F
    tmp_7 = F.embedding(in_0, in_5, 0, None, 2.0, False, False)
    tmp_8 = F.embedding(in_6, in_4, None, None, 2.0, False, False)
    tmp_9 = tmp_7 + tmp_8
    tmp_10 = F.embedding(in_7, in_3, None, None, 2.0, False, False)
    tmp_9 += tmp_10
    tmp_12 = F.dropout(tmp_9, 0.1, False, False)
    tmp_13 = F.layer_norm(tmp_12, (768,), in_2, in_1, 1e-12)
    return tmp_13


# ── Argument extractor ───────────────────────────────────────────────────────
def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    # (word_ids, tok_ids, pos_ids, word_w, tok_w, pos_w, ln_w, ln_b)
    return (in_0, in_6, in_7, in_5, in_4, in_3, in_2, in_1)


# ── Kernel wrapper ───────────────────────────────────────────────────────────
@torch.fx.wrap
def kernel_wrapper(word_ids, tok_ids, pos_ids, word_w, tok_w, pos_w, ln_w, ln_b):
    ln_out = run_fused_embed_add_ln(
        word_ids, tok_ids, pos_ids,
        word_w, tok_w, pos_w,
        ln_w, ln_b,
        D=_D, BLOCK_D=_BLOCK_D,
        store_emb=False,
    )
    return ln_out


# ── Replacement function (zero-argument, returns callable) ───────────────────
def replacement_func():
    return kernel_wrapper