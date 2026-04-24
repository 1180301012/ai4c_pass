import torch
from pass_dir.shared_kernels import dispatch


# ── Pattern ──────────────────────────────────────────────────────────────────
def pattern(in_2, in_1, in_0):
    return torch.nn.functional.linear(in_2, in_1, in_0)


# ── Routing: append "linear" so dispatch() knows which kernel calls which ──────
def replacement_args(in_2, in_1, in_0):
    return (in_2, in_1, in_0, "linear")


# ── Replacement entry-point ───────────────────────────────────────────────────
# Both passes return dispatch_v1 (same object from shared_kernels.py).
# replacement_func_limit counts unique names → 1 unique replacement func.
def replacement_func():
    return dispatch