"""
Fused pass for batched matmul via torch.matmul(in_1, in_0) followed by view.
Matches: tmp_0 = torch.matmul(in_1, in_0)
The Triton GEMV kernel (route "gemv") handles the N_out=1 case efficiently.
"""
import torch
from pass_dir.dispatch import dispatch


# ──────────────────────────────────────────────────────────────────────────────
# Pattern  (GCNet / S-ViPNAS: torch.matmul(in_1, in_0) with N_out=1)
# ──────────────────────────────────────────────────────────────────────────────
def pattern(in_0, in_1):
    tmp_0 = torch.matmul(in_1, in_0)
    return tmp_0


# ──────────────────────────────────────────────────────────────────────────────
# Argument extractor
# ──────────────────────────────────────────────────────────────────────────────
def replacement_args(in_0, in_1):
    return (in_0, in_1, "gemv")


# ──────────────────────────────────────────────────────────────────────────────
# Replacement factory – returns the SAME shared dispatch object as the other
# pass so that replacement_func_limit is never hit.
# ──────────────────────────────────────────────────────────────────────────────
def replacement_func():
    return dispatch