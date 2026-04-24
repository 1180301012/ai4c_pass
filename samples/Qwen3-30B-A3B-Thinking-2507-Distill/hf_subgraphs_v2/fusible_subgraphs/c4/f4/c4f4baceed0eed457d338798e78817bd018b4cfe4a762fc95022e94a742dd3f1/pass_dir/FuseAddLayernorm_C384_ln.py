import torch
from pass_dir.shared_layernorm import dispatch_layernorm


def pattern(x, weight, bias):
    """Fallback: match layer_norm with normalized_shape=(384,)."""
    out = torch.nn.functional.layer_norm(x, (384,), weight, bias, 1e-05)
    return out


def replacement_args(x, weight, bias):
    return (x, weight, bias, "C384")


def replacement_func():
    return dispatch_layernorm