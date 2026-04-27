import torch
from pass_dir._shared_matmul import dispatch_matmul  # shared replacement func


# ──────────────────────────────────────────────────────────────────────────────
# Pattern: match  torch.matmul(in_1, in_0)
# Used by GCNet, S-ViPNAS and float32 yolo models.
# ──────────────────────────────────────────────────────────────────────────────
def pattern(in_0, in_1):
    return torch.matmul(in_1, in_0)


def replacement_args(in_0, in_1):
    return (in_0, in_1, "torch_matmul")


def replacement_func():
    return dispatch_matmul