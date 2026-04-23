import torch
import triton
import triton.language as tl

# Simple layer_norm pattern first to verify matching works
def pattern(x, w, b):
    return torch.nn.functional.layer_norm(x, (1024,), w, b, 1e-05)

def replacement_args(x, w, b):
    return (x, w, b, "route_ln_1024")

def replacement_func():
    from pass_dir.FusedEmbedAddLayerNorm import dispatch_fused_layernorm
    return dispatch_fused_layernorm