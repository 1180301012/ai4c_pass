import torch
import triton
import triton.language as tl
from pass_dir._fused_kernel import dispatch_wrapper, fused_embed_add_ln_kernel


def pattern(a, b, c, d, e, f, g, h):
    embed1 = torch.nn.functional.embedding(a, f, 0, None, 2.0, False, False)
    embed2 = torch.nn.functional.embedding(c, d, None, None, 2.0, False, False)
    sum1 = embed1 + embed2
    embed3 = torch.nn.functional.embedding(e, b, None, None, 2.0, False, False)
    sum2 = sum1 + embed3
    dropout_out = torch.nn.functional.dropout(sum2, 0.1, False, False)
    ln_out = torch.nn.functional.layer_norm(dropout_out, (1024,), g, h, 1e-12)
    return (dropout_out, ln_out)


def replacement_args(a, b, c, d, e, f, g, h):
    # Order: input_ids, word_emb, token_type_ids, token_type_emb,
    #        position_ids, position_emb, ln_weight, ln_bias, route
    return (a, f, c, d, e, b, g, h, "route_1024_2out")


def replacement_func():
    return dispatch_wrapper