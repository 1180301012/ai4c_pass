import torch
from pass_dir._shared_matmul import dispatch_matmul  # shared replacement func


# ──────────────────────────────────────────────────────────────────────────────
# Pattern: match  in_1 @ in_0  (Python @ / __matmul__ operator)
# Used by yolo11 models (float16, bfloat16, float32 variants).
# ──────────────────────────────────────────────────────────────────────────────
def pattern(in_0, in_1):
    return in_1 @ in_0


def replacement_args(in_0, in_1):
    return (in_0, in_1, "at_op")


def replacement_func():
    return dispatch_matmul