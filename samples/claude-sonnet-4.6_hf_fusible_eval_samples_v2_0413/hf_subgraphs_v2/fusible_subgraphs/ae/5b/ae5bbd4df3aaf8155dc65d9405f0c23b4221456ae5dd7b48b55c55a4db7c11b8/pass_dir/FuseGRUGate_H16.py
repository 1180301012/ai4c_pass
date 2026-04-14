"""
Fused pass for WavLM GRU relative position gate computation (H=16 variant).

Fuses: linear → view(1,16,199,2,4) → sum(-1) → sigmoid → chunk →
       element-wise gates → view(1,16,-1,1)

Into a single Triton kernel via the shared implementation in gru_gate_impl.py.
Both H=12 and H=16 passes return the SAME fused_gru_gate function object,
satisfying the output_pass_replacement_func_limit constraint.
"""

import torch
from pass_dir.gru_gate_impl import fused_gru_gate  # shared dispatch function object


# ---------------------------------------------------------------------------
# Pattern: mirrors model.py exactly (H=16 variant)
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2, in_3):
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_4 = linear.view(1, 16, 199, 2, 4)
    tmp_5 = tmp_4.sum(-1, keepdim=False)
    tmp_6 = torch.sigmoid(tmp_5)
    chunk = tmp_6.chunk(2, dim=-1)
    tmp_8 = chunk[0]
    tmp_9 = chunk[1]
    tmp_10 = tmp_9 * in_2
    tmp_11 = tmp_10 - 1.0
    tmp_12 = tmp_8 * tmp_11
    tmp_13 = tmp_12 + 2.0
    tmp_14 = tmp_13.view(1, 16, -1, 1)
    return tmp_14


def replacement_args(in_0, in_1, in_2, in_3):
    # Append route tag so the shared dispatch knows which variant matched
    return (in_0, in_1, in_2, in_3, "h16")


def replacement_func():
    # Returns the SAME function object as FuseGRUGate_H12 — one unique func
    return fused_gru_gate