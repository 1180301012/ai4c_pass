import torch
import triton
import triton.language as tl
from pass_dir.shared_yolos_routes import yolos_dispatch


def pattern(in_6):
    tmp_28 = in_6[(slice(None, None, None), slice(None, None, None), slice(1, -10, None), slice(None, None, None))]
    tmp_29 = tmp_28.transpose(2, 3)
    tmp_30 = tmp_29.view(4, 32, 15, 15)
    tmp_32 = tmp_30.flatten(2)
    tmp_33 = tmp_32.transpose(1, 2)
    tmp_34 = tmp_33.contiguous()
    tmp_35 = tmp_34.view(4, 1, 225, 32)
    return tmp_35


def replacement_args(in_6):
    return (in_6[:, :, 1:-10, :], "copy225_clone_view")


def replacement_func():
    return yolos_dispatch