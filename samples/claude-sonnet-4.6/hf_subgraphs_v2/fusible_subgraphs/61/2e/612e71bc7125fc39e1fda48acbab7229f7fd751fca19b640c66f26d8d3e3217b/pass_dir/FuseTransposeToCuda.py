import torch
import triton
import triton.language as tl
from torch import device
from pass_dir.shared_kernels import dispatch_kernel, replacement_func  # noqa: F401


# ---------------------------------------------------------------------------
# Pattern: transpose then move to cuda (already on cuda → no-op move)
# Matches:
#   tmp_2 = in_0.t()
#   tmp_3 = tmp_2.to(device(type='cuda'))
# ---------------------------------------------------------------------------
def pattern(in_0):
    tmp_2 = in_0.t()
    tmp_3 = tmp_2.to(device(type='cuda'))
    return tmp_3


def replacement_args(in_0):
    # Route string tells dispatch_kernel to do a free view-based reshape
    return (in_0, "transpose")