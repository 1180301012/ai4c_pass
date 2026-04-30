import torch
from pass_dir._kernel_impl import fused_add_layernorm_dispatch


def pattern(in_0, in_1, in_2, in_3):
    """Match: add(in_3, in_2) + layer_norm with normalized_shape=(768,)
    
    This handles the reversed operand order (in_3 + in_2) found in some models.
    """
    tmp_2 = in_3 + in_2
    tmp_3 = torch.nn.functional.layer_norm(tmp_2, (768,), in_1, in_0, 1e-05)
    return tmp_3


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3, "route_768_rev")


def replacement_func():
    return fused_add_layernorm_dispatch