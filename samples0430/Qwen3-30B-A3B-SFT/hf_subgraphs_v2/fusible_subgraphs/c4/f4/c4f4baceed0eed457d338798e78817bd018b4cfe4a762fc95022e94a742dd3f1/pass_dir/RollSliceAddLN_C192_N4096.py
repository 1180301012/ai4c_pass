import torch
import triton
import triton.language as tl
from pass_dir.shared_roll_add_ln import ln_dispatch


def pattern(in_0, in_1, in_2):
    tmp_9 = torch.nn.functional.layer_norm(in_2, (192,), in_1, in_0, 1e-05)
    return tmp_9


def replacement_args(in_0, in_1, in_2):
    return (in_2, in_1, in_0, "c192")


def replacement_func():
    return ln_dispatch