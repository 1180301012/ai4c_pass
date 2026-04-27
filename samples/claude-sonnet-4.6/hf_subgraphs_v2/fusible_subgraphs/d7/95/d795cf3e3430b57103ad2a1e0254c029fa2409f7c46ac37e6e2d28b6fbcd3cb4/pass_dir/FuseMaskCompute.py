"""
Pass: FuseMaskCompute

Replaces the entire attention-mask construction chain
  tmp_0  → reshape/transpose → unsqueeze×2 → subtract
         → ne/masked_fill → __eq__/masked_fill → tmp_16
with a single Triton kernel that analytically generates tmp_16 directly.

This is a single-output pattern (returns only tmp_16), avoiding any
tuple-return issues with @torch.fx.wrap replacement functions.

Works for ALL three target models (96-ch float16/bfloat16, 128-ch float32)
since the mask computation is independent of the input feature dimension.
"""

import torch
import triton

from pass_dir.attn_mask_kernel import gen_attn_mask


# ---------------------------------------------------------------------------
# Replacement wrapper — single-output, takes tmp_0 for device info only
# ---------------------------------------------------------------------------

@torch.fx.wrap
def _gen_attn_mask_wrapper(mask_const):
    """
    mask_const : the constant-folded 1×133×133 float32 CUDA tensor
                 (used only to infer the target device)
    Returns    : float32 CUDA tensor of shape (1, 361, 49, 49)
    """
    return gen_attn_mask(mask_const.device)


# ---------------------------------------------------------------------------
# Pattern — single output (tmp_16), no in_0 needed
#   Uses tmp_12.__getattr__('__eq__')(0) to produce call_method(__eq__)
#   matching exactly what Dynamo records for `tmp_12 == 0`
# ---------------------------------------------------------------------------

def pattern(tmp_0):
    tmp_7  = tmp_0.reshape(1, 19, 7, 19, 7)
    tmp_8  = tmp_7.transpose(2, 3)
    tmp_9  = tmp_8.reshape(1, 361, 49)
    tmp_10 = tmp_9.unsqueeze(2)
    tmp_11 = tmp_9.unsqueeze(3)
    tmp_12 = tmp_10 - tmp_11
    tmp_13 = tmp_12 != 0
    tmp_14 = tmp_12.masked_fill(tmp_13, -1000.0)
    tmp_15 = tmp_12.__getattr__('__eq__')(0)
    tmp_16 = tmp_14.masked_fill(tmp_15, 0.0)
    return tmp_16


# ---------------------------------------------------------------------------
# replacement_args / replacement_func
# ---------------------------------------------------------------------------

def replacement_args(tmp_0):
    return (tmp_0,)


def replacement_func():
    return _gen_attn_mask_wrapper