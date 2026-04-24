"""
Fuse: dropout(training=False) + linear  ->  single Triton GEMM+bias kernel
Target: BigBird (bfloat16)
  dropout(in_2, 0.1, False, False) -- identity (training=False)
  linear(tmp_3, in_1, in_0)         -- GEMM [1,17,768] x [3072,768]^T + [3072]

Uses shared dispatch wrapper (routing technique) so replacement_func_limit is not hit.
"""
import torch
from pass_dir.shared_linear_kernel import _dispatch_linear


# ── Pattern ──────────────────────────────────────────────────────────────────

def pattern(in_0, in_1, in_2):
    tmp_3 = torch.nn.functional.dropout(in_2, 0.1, False, False)
    linear = torch.nn.functional.linear(tmp_3, in_1, in_0)
    return linear


def replacement_args(in_0, in_1, in_2):
    # Route "bf16" → dispatch to bfloat16 path
    return (in_0, in_1, in_2, "bf16")


def replacement_func():
    return _dispatch_linear