"""
Pass: FuseScaleClampDivMul_scale144

Matches the 4-op subgraph that follows the precomputed L2-norm:
    tmp_4 = tmp_3 * 0.14433756729740643   (scale the norm)
    tmp_5 = tmp_4.clamp(min=1e-05)        (clamp denominator)
    tmp_6 = tmp_2 / tmp_5                 (normalise input)
    tmp_7 = tmp_6 * in_0                  (apply scalar weight)

Inputs:
  in_0  – scalar weight [1]
  tmp_3 – L2-norm result [B, C, 1]  (output of torch.functional.norm)
  tmp_2 – flattened input [B, C, FEAT]
"""

import os, sys, torch, triton, triton.language as tl

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from kern_relu_norm import norm_dispatch   # shared function object → same id


def pattern(in_0, tmp_3, tmp_2):
    tmp_4 = tmp_3 * 0.14433756729740643
    tmp_5 = tmp_4.clamp(min=1e-05)
    tmp_6 = tmp_2 / tmp_5
    tmp_7 = tmp_6 * in_0
    return tmp_7


def replacement_args(in_0, tmp_3, tmp_2):
    return (in_0, tmp_3, tmp_2, "scale_144")


def replacement_func():
    return norm_dispatch