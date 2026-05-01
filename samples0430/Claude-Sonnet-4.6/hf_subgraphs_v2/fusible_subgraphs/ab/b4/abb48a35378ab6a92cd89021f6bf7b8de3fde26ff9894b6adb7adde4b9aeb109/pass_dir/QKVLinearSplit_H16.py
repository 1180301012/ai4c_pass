"""
Optimization pass: fused QKV linear + reshape + permute + split + K-transpose
for attention heads H=16 (convit_base variants).

Matches:
  linear(in_1, in_0)        -> [1, 197, 2304]
  .reshape(1,197,3,16,48)   -> [1, 197, 3, 16, 48]
  .permute(2,0,3,1,4)       -> [3, 1, 16, 197, 48]
  .unbind(0)                -> Q[1,16,197,48], K[1,16,197,48], V[1,16,197,48]
  K.transpose(-2,-1)        -> KT[1,16,48,197]
Returns: (Q, KT, V)

Uses the shared dispatch wrapper from qkv_kernels so replacement_func_limit
never drops this pass.
"""

import torch
from pass_dir.qkv_kernels import _qkv_replacement


# ---------------------------------------------------------------------------
# Pattern – must mirror model.py exactly (method calls, positional args)
# ---------------------------------------------------------------------------
def pattern(in_0, in_1):
    linear = torch.nn.functional.linear(in_1, in_0, None)
    tmp_2 = linear.reshape(1, 197, 3, 16, 48)
    tmp_3 = tmp_2.permute(2, 0, 3, 1, 4)
    unbind = tmp_3.unbind(0)
    tmp_5 = unbind[0]
    tmp_6 = unbind[1]
    tmp_7 = unbind[2]
    tmp_8 = tmp_6.transpose(-2, -1)
    return (tmp_5, tmp_8, tmp_7)


# ---------------------------------------------------------------------------
# Argument extraction
# ---------------------------------------------------------------------------
def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ---------------------------------------------------------------------------
# Return the shared replacement function (same object across all passes)
# ---------------------------------------------------------------------------
def replacement_func():
    return _qkv_replacement