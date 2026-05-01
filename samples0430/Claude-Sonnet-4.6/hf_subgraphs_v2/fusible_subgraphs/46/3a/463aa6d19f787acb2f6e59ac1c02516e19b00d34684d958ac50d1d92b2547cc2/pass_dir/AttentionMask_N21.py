import torch
import triton
import triton.language as tl
from pass_dir.shared_dispatch import _dispatch  # shared object → fixes replacement_func_limit


def pattern(data, eq_result):
    all_res = torch.all(eq_result, dim=-1, keepdim=True)
    not_res = ~all_res
    mul_res = data.mul(not_res)
    return (mul_res,)


def replacement_args(data, eq_result):
    return (data, eq_result, "TAIL")


def replacement_func():
    return _dispatch