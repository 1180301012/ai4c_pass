import torch
from pass_dir.trocr_decoder_embed_ln_shared import trocr_decoder_dispatch


def pattern(ids, tok_weight, addend, ln_bias, ln_weight):
    tmp_0 = torch.nn.functional.embedding(ids, tok_weight, 1, None, 2.0, False, False)
    tmp_1 = tmp_0 * 16.0
    tmp_2 = tmp_1 + addend
    out = torch.nn.functional.layer_norm(tmp_2, (256,), ln_weight, ln_bias, 1e-05)
    return out


def replacement_args(ids, tok_weight, addend, ln_bias, ln_weight):
    return (addend, tok_weight, ln_weight, ln_bias, ids, "embed_mul_add_ln")


def replacement_func():
    return trocr_decoder_dispatch