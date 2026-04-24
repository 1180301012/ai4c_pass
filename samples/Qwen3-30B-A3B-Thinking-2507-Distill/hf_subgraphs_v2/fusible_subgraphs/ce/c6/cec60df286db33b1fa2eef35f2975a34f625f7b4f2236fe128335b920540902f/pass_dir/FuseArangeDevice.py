import torch
import triton
import triton.language as tl
from torch import device
from pass_dir.shared_arange_dispatch import shared_arange_dispatch


def pattern(tmp_0):
    # torch.arange(0,1) is a constant - no device placeholder needed.
    # tmp_0 is the ONLY placeholder: it matches the arange output node in the target.
    # Returning the SINGLE arange result (not a placeholder) avoids circular-dependency crash.
    tmp_0 = torch.arange(0, 1, device='cuda:0')
    return tmp_0


def replacement_args(tmp_0):
    # Route "arange" → replacement returns _cached_arange[0] (pre-cached on first call)
    return ("arange",)


def replacement_func():
    return shared_arange_dispatch