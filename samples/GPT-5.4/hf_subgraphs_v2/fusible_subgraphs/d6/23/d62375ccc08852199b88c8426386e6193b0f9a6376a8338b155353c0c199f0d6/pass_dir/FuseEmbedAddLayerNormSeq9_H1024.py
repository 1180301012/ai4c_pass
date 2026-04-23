import torch
import triton
import triton.language as tl
from pass_dir.shared_fused_ops import shared_dispatch


# Match a robust normalized form: add followed by layer_norm, returning both
# the add result and normalized result. This captures the observable outputs.
def pattern(in_0, in_1, in_2, in_3):
    tmp_12 = in_0 + in_1
    tmp_14 = torch.nn.functional.layer_norm(tmp_12, (1024,), in_3, in_2, 1e-05)
    return (tmp_12, tmp_14)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3, "add_ln_h1024")


def replacement_func():
    return shared_dispatch