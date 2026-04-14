import torch
import triton
import triton.language as tl
from pass_dir.attn_kernel import fused_matmul_permute


# ── Pattern: matmul(attn_weights, values) → permute(0,2,1,3) → contiguous ──
# attn_weights is the dropout output (= softmax output since training=False).
# This single pattern covers ALL 6 graphs regardless of scale or reshape shape.

def pattern(tmp_3, in_3):
    matmul = torch.matmul(tmp_3, in_3)
    tmp_5 = matmul.permute(0, 2, 1, 3)
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(tmp_3, in_3):
    return (tmp_3, in_3)


def replacement_func():
    return fused_matmul_permute