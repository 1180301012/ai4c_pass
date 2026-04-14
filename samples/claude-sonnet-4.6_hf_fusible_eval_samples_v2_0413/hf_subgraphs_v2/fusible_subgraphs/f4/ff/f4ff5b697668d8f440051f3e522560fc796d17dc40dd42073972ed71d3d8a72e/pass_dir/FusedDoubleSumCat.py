import torch
import triton
import triton.language as tl
from pass_dir.kernel_impl import dispatch_fused


# B-agnostic fallback: match sum+sum+cat for graphs where mul+reshape can't match
def pattern(tmp_5, tmp_8):
    tmp_6 = torch.sum(tmp_5, dim=2, keepdim=True)
    tmp_9 = torch.sum(tmp_8, dim=2, keepdim=True)
    tmp_10 = torch.cat([tmp_6, tmp_9], dim=-1)
    return tmp_10


def replacement_args(tmp_5, tmp_8):
    return (tmp_5, tmp_8, "double_sum")


def replacement_func():
    return dispatch_fused