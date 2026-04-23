import torch
import triton
import triton.language as tl
from pass_dir.shared_fused_ops import shared_dispatch


def pattern(in_0, in_2, in_3):
    tmp_13 = torch.nn.functional.dropout(in_0, p=0.1, training=False)
    tmp_14 = torch.nn.functional.layer_norm(tmp_13, (1024,), in_3, in_2, 1e-05)
    return (tmp_13, tmp_14)


def replacement_args(in_0, in_2, in_3):
    return (in_0, in_2, in_3, "dropout_ln_h1024")


def replacement_func():
    return shared_dispatch