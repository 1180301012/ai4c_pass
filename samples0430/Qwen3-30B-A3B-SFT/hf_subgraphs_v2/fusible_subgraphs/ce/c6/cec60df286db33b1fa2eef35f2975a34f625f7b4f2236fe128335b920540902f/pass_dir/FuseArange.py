import torch
import triton
import triton.language as tl

from pass_dir.shared_dispatch import fused_dispatch


# Pass 0: Replace arange(0, 1) with a constant-folded dispatch.
# The model has no inputs so the arange output is always tensor([0]).
# tmp_0 is observable (used in the model's return), so we MUST include it.
# We expose it via the "route_arange_1d" branch in fused_dispatch.
def pattern():
    tmp_0 = torch.arange(0, 1, device='cuda:0')
    tmp_1 = tmp_0.unsqueeze(0)
    tmp_2 = tmp_1.repeat(1, 1)
    return tmp_0, tmp_2


def replacement_args():
    return (None, "route_arange_1d")


def replacement_func():
    return fused_dispatch