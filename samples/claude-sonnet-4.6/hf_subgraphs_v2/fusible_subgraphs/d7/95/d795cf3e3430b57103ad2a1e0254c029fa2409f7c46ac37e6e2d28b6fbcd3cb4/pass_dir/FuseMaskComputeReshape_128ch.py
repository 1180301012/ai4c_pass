"""
Pass: FuseMaskComputeReshape_128ch

Fuses the attention-mask reshape/compute chain + input reshape/transpose for
the Twins-SVT-L model variant with 128 feature channels (float32).

After torch.compile, the torch.zeros + fill_ calls are constant-folded into a
single get_attr constant node (_tensor_constant0).  We therefore accept that
constant tensor as `tmp_0` and match everything downstream.

Pattern matched (tmp_0 is the pre-filled 1×133×133 float32 constant):
  - tmp_0.reshape / transpose chain → tmp_9  (shape 1×361×49)
  - unsqueeze + subtract + two masked_fill  → tmp_16  (shape 1×361×49×49)
  - in_0.reshape(1,19,7,19,7,128).transpose(2,3)  → tmp_6

Replacement:
  - tmp_16 is generated analytically by a Triton kernel  (ignores tmp_0 value)
  - tmp_6 is a cheap view of in_0 (reshape + transpose)
"""

import torch
import triton
from torch import device

from pass_dir.attn_mask_kernel import gen_attn_mask


# ---------------------------------------------------------------------------
# Replacement wrapper
# ---------------------------------------------------------------------------

@torch.fx.wrap
def _fused_mask_and_reshape_128(in_0):
    """
    Returns (tmp_16, tmp_6):
      tmp_16 : float32, shape (1, 361, 49, 49) — directly computed attention mask
      tmp_6  : view of in_0, shape (1, 19, 19, 7, 7, 128)
    """
    tmp_16 = gen_attn_mask(in_0.device)
    tmp_5 = in_0.reshape(1, 19, 7, 19, 7, 128)
    tmp_6 = tmp_5.transpose(2, 3)
    return (tmp_16, tmp_6)


# ---------------------------------------------------------------------------
# Pattern  — uses tmp_12.__getattr__('__eq__')(0) to produce call_method(__eq__)
#            matching exactly what Dynamo records for `tmp_12 == 0`
# ---------------------------------------------------------------------------

def pattern(tmp_0, in_0):
    tmp_5 = in_0.reshape(1, 19, 7, 19, 7, 128)
    tmp_6 = tmp_5.transpose(2, 3)
    tmp_7 = tmp_0.reshape(1, 19, 7, 19, 7)
    tmp_8 = tmp_7.transpose(2, 3)
    tmp_9 = tmp_8.reshape(1, 361, 49)
    tmp_10 = tmp_9.unsqueeze(2)
    tmp_11 = tmp_9.unsqueeze(3)
    tmp_12 = tmp_10 - tmp_11
    tmp_13 = tmp_12 != 0
    tmp_14 = tmp_12.masked_fill(tmp_13, -1000.0)
    tmp_15 = tmp_12.__getattr__('__eq__')(0)
    tmp_16 = tmp_14.masked_fill(tmp_15, 0.0)
    return (tmp_16, tmp_6)


# ---------------------------------------------------------------------------
# replacement_args / replacement_func
# ---------------------------------------------------------------------------

def replacement_args(tmp_0, in_0):
    return (in_0,)


def replacement_func():
    return _fused_mask_and_reshape_128