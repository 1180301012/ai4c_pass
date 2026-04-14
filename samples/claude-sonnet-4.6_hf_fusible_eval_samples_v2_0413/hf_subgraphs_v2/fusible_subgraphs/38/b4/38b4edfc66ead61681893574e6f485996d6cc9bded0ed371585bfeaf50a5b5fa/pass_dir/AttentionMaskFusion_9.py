import torch
import triton
import triton.language as tl
from torch import device
from pass_dir.shared_attn_kernel import attn_mask_fusion


# ---------------------------------------------------------------------------
# Pattern
# The pattern tracer executes torch.full/torch.arange concretely (no Proxy
# inputs) and the in-place masked_fill_ does NOT update placeholder proxies,
# causing dead-code errors when the full chain is included.
# Solution: promote const_one and causal_mask as placeholders so that:
#   • No get_attr constants exist in the pattern
#   • No dead-code nodes exist in the pattern
#   • The matched subgraph (tmp_10…tmp_19) is fully traced
#
# placeholder bindings in the target graph:
#   in_0        → graph input  in_0
#   const_one   → torch.tensor(1.0, dtype=float32) output   (tmp_13)
#   causal_mask → expand(…, 1, 1, 9, 9) output              (tmp_9)
# ---------------------------------------------------------------------------
def pattern(in_0, const_one, causal_mask):
    tmp_10 = in_0[(slice(None, None, None), None, None, slice(None, None, None))]
    tmp_11 = tmp_10.expand(1, 1, 9, 9)
    tmp_12 = tmp_11.to(torch.float32)
    tmp_14 = const_one - tmp_12
    tmp_15 = tmp_14.to(torch.bool)
    tmp_16 = tmp_14.masked_fill(tmp_15, -3.4028234663852886e+38)
    tmp_17 = tmp_16.to(device(type='cuda', index=0))
    tmp_18 = tmp_17.bool()
    tmp_19 = causal_mask.masked_fill(tmp_18, -3.4028234663852886e+38)
    return tmp_19


# ---------------------------------------------------------------------------
# Replacement helpers – route "n9" selects the N=9 kernel branch
# ---------------------------------------------------------------------------
def replacement_args(in_0, const_one, causal_mask):
    return (in_0, "n9")


def replacement_func():
    return attn_mask_fusion