import torch
from pass_dir._shared_kernels import fused_dispatch


# ── Pass interface ────────────────────────────────────────────────────────────
# Pattern: add + dropout(training=False) + layer_norm with H=2048
def pattern(x, y, weight, bias):
    add     = x + y
    dropped = torch.nn.functional.dropout(add, p=0.1, training=False)
    ln      = torch.nn.functional.layer_norm(dropped, (2048,), weight, bias, 1e-05)
    return (dropped, ln)


def replacement_args(x, y, weight, bias):
    return (x, y, weight, bias, '2048')


def replacement_func():
    return fused_dispatch