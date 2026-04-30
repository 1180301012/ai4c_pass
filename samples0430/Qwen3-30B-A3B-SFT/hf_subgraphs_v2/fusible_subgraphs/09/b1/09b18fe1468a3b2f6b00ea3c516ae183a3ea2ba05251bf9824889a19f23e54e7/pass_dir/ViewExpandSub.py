import torch
import triton
import triton.language as tl

# Import the SAME dispatch function object — identical replacement_func() satisfies
# output_pass_replacement_func_limit=1.
from pass_dir.FusedDistSoftmaxViewExpandSub import _fused_dispatch


# ── pattern: unsqueeze + expand + view + sub (5 inputs, aten ops) ─────────────
def pattern(in_0, in_1, in_2, in_3, in_4):
    tmp_7 = torch.ops.aten.unsqueeze.default(in_4, 2)
    tmp_8 = torch.ops.aten.expand.default(tmp_7, [1, 4096, 32, 512])
    tmp_6 = torch.ops.aten.view.default(in_0, [1, 1, 32, 512])
    tmp_10 = torch.ops.aten.sub.Tensor(tmp_8, tmp_6)
    return tmp_10


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4, "view_expand_sub")


def replacement_func():
    return _fused_dispatch