import torch
from pass_dir._shared_gemm import dispatch_matmul  # noqa: F401


# ──────────────────────────────────────────────────────────────────────────────
# Pattern: `in_1 @ in_0`  (operator.matmul – used in YOLO model files)
# ──────────────────────────────────────────────────────────────────────────────

def pattern(in_0, in_1):
    result = in_1 @ in_0
    return result


def replacement_args(in_0, in_1):
    return (in_0, in_1, "at")


# ──────────────────────────────────────────────────────────────────────────────
# Required by the AI4C framework.
# Returns the SAME dispatch_matmul object as FuseMatmulView_TorchMatmul.py so
# both passes share one unique replacement_func and bypass the func-limit.
# ──────────────────────────────────────────────────────────────────────────────

def replacement_func():
    return dispatch_matmul