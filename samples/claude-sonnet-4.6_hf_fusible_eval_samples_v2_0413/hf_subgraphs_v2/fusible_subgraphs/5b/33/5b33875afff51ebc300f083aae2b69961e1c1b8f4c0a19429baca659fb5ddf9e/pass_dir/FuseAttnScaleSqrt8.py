import torch
import triton
import triton.language as tl
from pass_dir.attn_kernel import fused_dispatch_simple


# ── Pattern: scale=√8 — covers tiny-random-ConvBert ──
# The reshape(in_1, ...) is NOT in this pattern; it stays in the graph unchanged.

def pattern(in_0, in_2, in_3):
    tmp_0 = in_0 / 2.8284271247461903
    tmp_1 = tmp_0 + in_2
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.1, False, False)
    matmul = torch.matmul(tmp_3, in_3)
    tmp_5 = matmul.permute(0, 2, 1, 3)
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(in_0, in_2, in_3):
    return (in_0, in_2, in_3, "scale_sqrt8")


def replacement_func():
    return fused_dispatch_simple