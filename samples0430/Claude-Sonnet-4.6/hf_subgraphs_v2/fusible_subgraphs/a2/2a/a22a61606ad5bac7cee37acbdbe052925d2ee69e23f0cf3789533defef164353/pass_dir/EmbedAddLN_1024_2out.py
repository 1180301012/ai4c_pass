import torch
import triton
import triton.language as tl
from pass_dir.embed_ln_kernels import dispatch_embed_add_ln


# Pattern: 3 embedding lookups + 2 adds → fused embedding sum
# Uses F.embedding (high-level op as in original GraphModule)
# Try both out-of-place add variants to match either operator.add or operator.iadd

def pattern(word_ids, word_emb, tt_ids, tt_emb, pos_ids, pos_emb):
    e_word = torch.nn.functional.embedding(word_ids, word_emb, 0, None, 2.0, False, False)
    e_tt   = torch.nn.functional.embedding(tt_ids,   tt_emb,  None, None, 2.0, False, False)
    s      = e_word + e_tt
    e_pos  = torch.nn.functional.embedding(pos_ids,  pos_emb, None, None, 2.0, False, False)
    s      = s + e_pos
    return s


def replacement_args(word_ids, word_emb, tt_ids, tt_emb, pos_ids, pos_emb):
    return (word_emb, word_ids, tt_emb, tt_ids, pos_emb, pos_ids, "emb")


def replacement_func():
    return dispatch_embed_add_ln