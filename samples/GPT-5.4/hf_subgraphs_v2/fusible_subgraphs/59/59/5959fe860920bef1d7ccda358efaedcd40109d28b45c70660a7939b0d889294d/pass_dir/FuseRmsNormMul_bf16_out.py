import torch
import triton
import triton.language as tl

from pass_dir.shared_rmsnorm import fused_dispatch


def pattern(in_0: torch.Tensor, in_2: torch.Tensor):
    tmp_10 = in_2.to(torch.float32)
    tmp_11 = tmp_10.pow(2)
    tmp_12 = tmp_11.mean(-1, keepdim=True)
    tmp_13 = tmp_12 + 1e-06
    tmp_14 = torch.rsqrt(tmp_13)
    tmp_15 = tmp_10 * tmp_14
    tmp_16 = tmp_15.to(torch.bfloat16)
    tmp_17 = in_0 * tmp_16
    return tmp_17


def replacement_args(in_0: torch.Tensor, in_2: torch.Tensor):
    return (in_0, in_2, "smollm_bf16")


def replacement_func():
    return fused_dispatch