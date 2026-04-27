import torch
import triton
import triton.language as tl
from torch import device

from pass_dir.shared_dispatch import dispatch_wrapper


# ---------------------------------------------------------------------------
# Pattern: matches the torch.arange(0, 1, device=cuda) factory call.
#
#   In model.py:
#     tmp_0 = torch.arange(0, 1, device = device(type='cuda', index=0))
#
#   Zero-arg pattern: no placeholder nodes.  The framework matches the exact
#   call_function node and replaces it with dispatch_wrapper("route_arange").
# ---------------------------------------------------------------------------
def pattern():
    tmp_0 = torch.arange(0, 1, device=device(type='cuda', index=0))
    return tmp_0


def replacement_args():
    return ("route_arange",)


def replacement_func():
    return dispatch_wrapper