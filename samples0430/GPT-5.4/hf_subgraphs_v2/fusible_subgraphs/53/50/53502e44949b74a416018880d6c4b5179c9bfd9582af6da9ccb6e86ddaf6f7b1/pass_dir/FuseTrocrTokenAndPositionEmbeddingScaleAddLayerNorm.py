import torch
from pass_dir.trocr_decoder_embed_ln_shared import trocr_decoder_dispatch


def pattern(tok_ids, tok_weight, pos_ids, pos_weight, ln_bias, ln_weight):
    tmp_0 = torch.nn.functional.embedding(tok_ids, tok_weight, 1, None, 2.0, False, False)
    tmp_1 = tmp_0 * 16.0
    tmp_2 = torch.nn.functional.embedding(pos_ids, pos_weight, None, None, 2.0, False, False)
    tmp_3 = tmp_1 + tmp_2
    out = torch.nn.functional.layer_norm(tmp_3, (256,), ln_weight, ln_bias, 1e-05)
    return out


def replacement_args(tok_ids, tok_weight, pos_ids, pos_weight, ln_bias, ln_weight):
    return (tok_weight, pos_weight, ln_weight, ln_bias, tok_ids, pos_ids, "two_embed_mul_add_ln")


def replacement_func():
    return trocr_decoder_dispatch