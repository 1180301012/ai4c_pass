import torch
from pass_dir.shared_dispatch import dispatch


# Fallback: fuse layer_norm + dropout (training=False → identity) for cases
# where the full embedding-sum pattern didn't match.
def pattern(x, weight, bias):
    normed = torch.nn.functional.layer_norm(x, (768,), weight, bias, 1e-12)
    out = torch.nn.functional.dropout(normed, 0.1, False, False)
    return out


def replacement_args(x, weight, bias):
    # 11 args: x, weight, bias, 3 dummies + route tag
    return (x, weight, bias, weight, weight, weight, weight, weight, weight, weight, "layernorm")


def replacement_func():
    return dispatch