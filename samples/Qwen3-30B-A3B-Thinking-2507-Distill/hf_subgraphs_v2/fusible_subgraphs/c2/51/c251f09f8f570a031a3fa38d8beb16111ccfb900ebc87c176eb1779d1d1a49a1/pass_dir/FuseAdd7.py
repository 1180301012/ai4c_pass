import torch
from pass_dir.shared_dispatch import dispatch


# Full pattern: embedding-sum + layer_norm + dropout
# This is unique in the graph, avoiding false matches on isolated add pairs.
def pattern(word_emb, pos_emb, emb_x1, emb_y1, emb_x2, emb_y2, diff_y, diff_x, weight, bias):
    t1 = word_emb + pos_emb
    t2 = t1 + emb_x1
    t3 = t2 + emb_y1
    t4 = t3 + emb_x2
    t5 = t4 + emb_y2
    t6 = t5 + diff_y
    t7 = t6 + diff_x
    normed = torch.nn.functional.layer_norm(t7, (768,), weight, bias, 1e-12)
    out = torch.nn.functional.dropout(normed, 0.1, False, False)
    return out


def replacement_args(word_emb, pos_emb, emb_x1, emb_y1, emb_x2, emb_y2, diff_y, diff_x, weight, bias):
    # route "full": 8 embeddings + weight + bias → fused kernel + layernorm
    # 11 args total: 10 tensors + route string so dispatch() gets all 11 args
    return (word_emb, pos_emb, emb_x1, emb_y1, emb_x2, emb_y2, diff_y, diff_x, weight, bias, "full")


def replacement_func():
    return dispatch