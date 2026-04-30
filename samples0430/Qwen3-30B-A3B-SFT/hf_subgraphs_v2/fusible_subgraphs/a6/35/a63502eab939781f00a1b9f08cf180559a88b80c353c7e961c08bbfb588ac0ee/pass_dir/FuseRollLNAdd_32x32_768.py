import torch
import triton
import triton.language as tl
from pass_dir.shared_ln_add_kernel import dispatch_layernorm_add


def pattern(in_0, in_1, in_2, tmp_5):
    """
    Match layer_norm(768) + residual add.
    tmp_5 is the roll-view [1, 1024, 768] passed through a view — it is
    already contiguous, so the kernel accesses it with a simple row stride.
    """
    tmp_6 = torch.nn.functional.layer_norm(tmp_5, (768,), in_1, in_0, 1e-05)
    tmp_7 = in_2 + tmp_6
    return (tmp_7,)


def replacement_args(in_0, in_1, in_2, tmp_5):
    return (in_0, in_1, in_2, tmp_5, "s32_c768")


def replacement_func():
    return dispatch_layernorm_add