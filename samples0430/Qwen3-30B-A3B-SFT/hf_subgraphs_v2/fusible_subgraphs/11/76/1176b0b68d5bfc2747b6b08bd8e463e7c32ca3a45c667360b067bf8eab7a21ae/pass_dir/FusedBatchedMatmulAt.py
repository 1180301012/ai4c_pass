"""
Fused pass for batched matmul via @ operator followed by view.
Matches: result = in_1 @ in_0
The Triton GEMM kernel computes the matmul; downstream .view() reshapes it.
"""
import torch
from pass_dir.dispatch import dispatch


# ──────────────────────────────────────────────────────────────────────────────
# Pattern  (yolo models: in_1 @ in_0 where shapes are [B,H,D,N] @ [B,H,N,N])
# ──────────────────────────────────────────────────────────────────────────────
def pattern(in_0, in_1):
    matmul = in_1 @ in_0
    return matmul


# ──────────────────────────────────────────────────────────────────────────────
# Argument extractor
# ──────────────────────────────────────────────────────────────────────────────
def replacement_args(in_0, in_1):
    return (in_0, in_1, "gemm")


# ──────────────────────────────────────────────────────────────────────────────
# Replacement factory – returns the SAME shared dispatch object as the other
# pass so that replacement_func_limit is never hit.
# ──────────────────────────────────────────────────────────────────────────────
def replacement_func():
    return dispatch