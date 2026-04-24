import torch
import triton
import triton.language as tl
from pass_dir.FusedEmbeddingLayerNorm_kernel import dispatch_layer_norm


def pattern(x, w, b):
    return torch.nn.functional.layer_norm(x, (32,), w, b, 1e-12)

def replacement_args(x, w, b):
    return (x, w, b, "ln_32")

def replacement_func():
    return dispatch_layer_norm