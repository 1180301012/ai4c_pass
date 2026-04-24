import torch
from pass_dir.shared_add_layernorm import dispatch_layernorm


def pattern(in_0, in_1, tmp_3):
    """Match only the layer_norm (single output)."""
    tmp_4 = torch.nn.functional.layer_norm(tmp_3, (768,), in_1, in_0, 1e-05)
    return tmp_4


def replacement_args(in_0, in_1, tmp_3):
    return (in_0, in_1, tmp_3, "768")


def replacement_func():
    return dispatch_layernorm