import torch
import triton
import triton.language as tl
from torch import device

# Import the shared dispatch function so all passes return the SAME object
# (required for replacement_func_limit to load all passes)
from pass_dir.arange_shared import dispatch_arange


# ── Pattern: aten.arange.default(1, ...) ─────────────────────────────────────
# pattern() is exempt from API validation; torch.ops.aten.* is allowed here.
def pattern():
    tmp_0 = torch.ops.aten.arange.default(
        1,
        device=device(type='cuda', index=0),
    )
    return tmp_0


def replacement_args():
    # Route string appended so all passes share the same replacement_func
    return ("aten_default",)


def replacement_func():
    return dispatch_arange