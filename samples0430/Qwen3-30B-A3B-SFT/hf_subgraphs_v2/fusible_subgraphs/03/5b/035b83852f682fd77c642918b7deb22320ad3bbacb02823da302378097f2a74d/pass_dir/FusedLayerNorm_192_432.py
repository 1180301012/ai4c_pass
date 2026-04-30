import torch
import triton
import triton.language as tl
from pass_dir.shared_ln_kernel import shared_layer_norm_dispatch


# ---------------------------------------------------------------------------
# Pass interface — handles layer_norm with normalized_shape = (192,)
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2):
    """Matches:  torch.nn.functional.layer_norm(in_2, (192,), in_1, in_0, 1e-06)"""
    tmp_2 = torch.nn.functional.layer_norm(in_2, (192,), in_1, in_0, 1e-06)
    return tmp_2


def replacement_args(in_0, in_1, in_2):
    # in_0=bias, in_1=weight, in_2=input; append route string for dispatch
    return (in_0, in_1, in_2, "ln_192")


def replacement_func():
    return shared_layer_norm_dispatch