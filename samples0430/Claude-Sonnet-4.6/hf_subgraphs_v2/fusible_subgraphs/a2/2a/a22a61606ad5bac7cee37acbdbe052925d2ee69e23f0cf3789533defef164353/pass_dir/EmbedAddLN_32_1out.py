import torch
import triton
import triton.language as tl
from pass_dir.embed_ln_kernels import dispatch_embed_add_ln


def pattern(x, ln_w, ln_b):
    out = torch.nn.functional.layer_norm(x, (32,), ln_w, ln_b, 1e-12)
    return out


def replacement_args(x, ln_w, ln_b):
    return (x, ln_w, ln_b, "ln_32")


def replacement_func():
    return dispatch_embed_add_ln