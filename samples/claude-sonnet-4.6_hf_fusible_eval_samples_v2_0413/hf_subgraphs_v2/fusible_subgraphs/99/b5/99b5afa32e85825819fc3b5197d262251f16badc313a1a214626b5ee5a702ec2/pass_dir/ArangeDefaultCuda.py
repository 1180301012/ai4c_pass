import torch
import triton
import triton.language as tl
from torch import device

from pass_dir.arange_shared import dispatch_arange


# ── Pattern: aten.arange.start_step(0, 1, 1, ...) ────────────────────────────
# pattern() is exempt from API validation; torch.ops.aten.* is allowed here.
def pattern():
    tmp_0 = torch.ops.aten.arange.start_step(
        0, 1, 1,
        device=device(type='cuda', index=0),
    )
    return tmp_0


def replacement_args():
    return ("aten_step",)


def replacement_func():
    return dispatch_arange