"""
PosBias_196 pass: replaces the deterministic (1,196,196,3) position-bias
computation with a precomputed constant, completely eliminating all the
CPU arange/repeat/repeat_interleave/setitem overhead on every forward call.

Uses the same shared triton_ln_dispatch function (route="pos_bias") so that
the replacement_func_limit is never exceeded.
"""
import torch
from pass_dir.ln_kernels import triton_ln_dispatch


# ── Pattern: pure value-producing chain, NO setitem, returns (tmp_15, tmp_17, tmp_19) ──
def pattern():
    tmp_4 = torch.arange(14)
    tmp_5 = tmp_4.view(1, -1)
    tmp_6 = torch.arange(14)
    tmp_7 = tmp_6.view(-1, 1)
    tmp_8 = tmp_5 - tmp_7
    tmp_9 = tmp_8.repeat(14, 14)
    tmp_10 = tmp_8.repeat_interleave(14, dim=0)
    tmp_11 = tmp_10.repeat_interleave(14, dim=1)
    tmp_12 = tmp_9 ** 2
    tmp_13 = tmp_11 ** 2
    tmp_14 = tmp_12 + tmp_13
    tmp_15 = tmp_14.unsqueeze(0)
    tmp_17 = tmp_11.unsqueeze(0)
    tmp_19 = tmp_9.unsqueeze(0)
    return tmp_15, tmp_17, tmp_19


# ── No tensor inputs; return precomputed triple via shared dispatcher ─────────
def replacement_args():
    return (None, None, None, "pos_bias_triple")


def replacement_func():
    return triton_ln_dispatch