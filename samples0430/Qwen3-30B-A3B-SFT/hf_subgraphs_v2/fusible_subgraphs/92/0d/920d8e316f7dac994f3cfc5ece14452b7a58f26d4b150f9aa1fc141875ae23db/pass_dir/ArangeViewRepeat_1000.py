import torch
import triton
import triton.language as tl
from torch import device
from pass_dir.shared_arange_repeat import dispatch_view_repeat


# ---------------------------------------------------------------------------
# Pattern: view(1, -1) → repeat(2, 1)
# Placeholder x binds to the arange output in the model graph.
# Matches both GAE float32 and GAE float16 (N=1000) subgraphs.
# ---------------------------------------------------------------------------

def pattern(x):
    tmp_1 = x.view(1, -1)
    tmp_2 = tmp_1.repeat(2, 1)
    return tmp_2


def replacement_args(x):
    return (x,)


def replacement_func():
    return dispatch_view_repeat