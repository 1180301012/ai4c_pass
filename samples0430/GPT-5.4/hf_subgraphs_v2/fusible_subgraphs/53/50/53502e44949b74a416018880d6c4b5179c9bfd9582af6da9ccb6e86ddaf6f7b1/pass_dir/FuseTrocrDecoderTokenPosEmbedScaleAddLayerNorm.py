import torch
from pass_dir.trocr_decoder_embed_ln_shared import trocr_decoder_dispatch


def pattern(x, addend, ln_bias, ln_weight):
    tmp = x + addend
    out = torch.nn.functional.layer_norm(tmp, (256,), ln_weight, ln_bias, 1e-05)
    return out


def replacement_args(x, addend, ln_bias, ln_weight):
    return (x, addend, ln_weight, ln_bias, "add_ln")


def replacement_func():
    return trocr_decoder_dispatch