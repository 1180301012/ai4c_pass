import torch
import triton
import triton.language as tl
from pass_dir.shared_yolos_routes import yolos_dispatch


def pattern(tmp_33):
    tmp_34 = tmp_33.contiguous()
    tmp_35 = tmp_34.view(4, 1, 225, 32)
    return tmp_35


def replacement_args(tmp_33):
    return (tmp_33, "copy225_contiguous_view")


def replacement_func():
    return yolos_dispatch