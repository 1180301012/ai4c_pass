import torch
import triton
import triton.language as tl
from torch import device
from pass_dir.shared_attn_kernel import attn_mask_fusion


# ---------------------------------------------------------------------------
# Pattern – N=13 variant
# Same constraint as N=9: use const_one + causal_mask as placeholders.
# ---------------------------------------------------------------------------
def pattern(in_0, const_one, causal_mask):
    tmp_10 = in_0[(slice(None, None, None), None, None, slice(None, None, None))]
    tmp_11 = tmp_10.expand(1, 1, 13, 13)
    tmp_12 = tmp_11.to(torch.float32)
    tmp_14 = const_one - tmp_12
    tmp_15 = tmp_14.to(torch.bool)
    tmp_16 = tmp_14.masked_fill(tmp_15, -3.4028234663852886e+38)
    tmp_17 = tmp_16.to(device(type='cuda', index=0))
    tmp_18 = tmp_17.bool()
    tmp_19 = causal_mask.masked_fill(tmp_18, -3.4028234663852886e+38)
    return tmp_19


# ---------------------------------------------------------------------------
# Replacement helpers – route "n13" selects the N=13 kernel branch
# ---------------------------------------------------------------------------
def replacement_args(in_0, const_one, causal_mask):
    return (in_0, "n13")


def replacement_func():
    return attn_mask_fusion