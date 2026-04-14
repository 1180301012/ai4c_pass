"""
Pass: FuseCombined
Matches BOTH independent computation chains simultaneously:
  • (in_1-in_2).pow(2).sum(dim=3) * in_3  → tmp_4  [B,I,K]
  • in_4.unsqueeze(2).expand(...) - in_0.view(...)  → tmp_10 [B,I,K,F]

Returns (tmp_4, tmp_10) together, which allows a SINGLE combined kernel
launch (Grid I=4096) that interleaves in_1 HBM reads with es_out HBM writes
→ bidirectional bandwidth utilisation.

Falls back to separate FuseDistanceSoftmax + FuseExpandSubtract passes if
the FX matcher doesn't support disconnected multi-output patterns.
"""

import torch
from pass_dir.shared_kernels import shared_fused_kernel


def pattern(in_0, in_1, in_2, in_3, in_4):
    # Distance-scale chain
    tmp_1 = in_1 - in_2
    tmp_2 = tmp_1.pow(2)
    tmp_3 = tmp_2.sum(dim=3)
    tmp_4 = in_3 * tmp_3
    # Expand-subtract chain
    tmp_6 = in_0.view((1, 1, 32, 512))
    tmp_7 = in_4.unsqueeze(2)
    tmp_8 = tmp_7.expand((1, 4096, 32, 512))
    tmp_10 = tmp_8 - tmp_6
    return (tmp_4, tmp_10)


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    # route='combined': a=in_0, b=in_1, c=in_2, d=in_3, e=in_4
    return (in_0, in_1, in_2, in_3, in_4, "combined")


def replacement_func():
    return shared_fused_kernel