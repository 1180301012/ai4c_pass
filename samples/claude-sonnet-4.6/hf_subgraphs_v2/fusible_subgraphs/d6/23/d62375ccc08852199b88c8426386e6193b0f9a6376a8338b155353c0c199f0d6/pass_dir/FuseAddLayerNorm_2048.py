import torch
from pass_dir._shared_kernels import fused_dispatch


# ── Pass interface ────────────────────────────────────────────────────────────
# Pattern: add + layer_norm (no dropout node) with H=2048
# Covers the case where dropout(training=False) is elided from the graph.
def pattern(x, y, weight, bias):
    add = x + y
    ln  = torch.nn.functional.layer_norm(add, (2048,), weight, bias, 1e-05)
    return (add, ln)


def replacement_args(x, y, weight, bias):
    return (x, y, weight, bias, '2048')


def replacement_func():
    return fused_dispatch