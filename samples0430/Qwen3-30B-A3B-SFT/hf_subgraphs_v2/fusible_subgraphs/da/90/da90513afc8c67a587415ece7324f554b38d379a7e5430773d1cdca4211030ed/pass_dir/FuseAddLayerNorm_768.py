import torch
import triton
import triton.language as tl
from pass_dir.shared_add_ln import dispatch_fused_add_ln  # noqa: F401 — shared object


# stub block — kernel logic lives in pass_dir/shared_add_ln
def _stub():
    pass


def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = in_2 + in_3
    tmp_3 = torch.nn.functional.layer_norm(tmp_2, (768,), in_1, in_0, 1e-05)
    return tmp_3


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3, "768")


def replacement_func():
    return dispatch_fused_add_ln