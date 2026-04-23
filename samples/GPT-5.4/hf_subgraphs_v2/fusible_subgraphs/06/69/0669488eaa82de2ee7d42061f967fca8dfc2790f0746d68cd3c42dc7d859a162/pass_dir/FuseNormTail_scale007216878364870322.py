import torch
from pass_dir.shared_tail_kernel import fused_tail_dispatch


def pattern(in_0, in_1):
    tmp_3 = torch.functional.norm(in_1, dim=-1, keepdim=True)
    tmp_4 = tmp_3 * 0.07216878364870322
    tmp_5 = tmp_4.clamp(min=1e-05)
    tmp_6 = in_1 / tmp_5
    tmp_7 = tmp_6 * in_0
    return (tmp_7,)


def replacement_args(in_0, in_1):
    return (in_1, torch.functional.norm(in_1, dim=-1, keepdim=True), in_0, "tail_scale_007216878364870322")


def replacement_func():
    return fused_tail_dispatch