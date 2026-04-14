import torch
import triton
import triton.language as tl
from pass_dir.unfold_kernels import dispatch_unfold_perm_reshape


# ---------------------------------------------------------------------------
# Fallback: cat([x, y, z], dim=0) + to(dtype=float16)
# Applied when the full-chain pass doesn't match.
# ---------------------------------------------------------------------------
def pattern(x, y, z):
    tmp_6 = torch.cat([x, y, z], dim=0)
    tmp_7 = tmp_6.to(dtype=torch.float16)
    return tmp_7


def replacement_args(x, y, z):
    return (x, "cat_to_f16", y, z)


def replacement_func():
    return dispatch_unfold_perm_reshape