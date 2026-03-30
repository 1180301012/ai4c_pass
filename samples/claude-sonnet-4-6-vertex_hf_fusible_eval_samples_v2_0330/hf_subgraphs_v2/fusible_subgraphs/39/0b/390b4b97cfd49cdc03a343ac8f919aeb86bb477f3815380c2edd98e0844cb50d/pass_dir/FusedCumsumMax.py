import torch
import triton
import triton.language as tl
from torch import device


# ─────────────────────────────────────────────────────────────────────────────
# Pattern: start at tmp_2 as a PLACEHOLDER so its external user (dead-code
# masked_fill_ node) does not trigger NOT_CONTAINED.
#
#   tmp_2 (placeholder = cumsum(in_1,-1) - 1)
#     │
#     ├─ unsqueeze(0) → expand(3,-1,-1) → to(cuda:0)  ──────────── out1
#     └─ max(0)[0] → max(-1,keepdim=True)[0] → +1 → -9 ─────────── out0
#
# Since expand(3) creates 3 identical stride-0 slices, max(dim=0)==tmp_2,
# so the chain collapses to:
#   out0 = tmp_2.amax(-1, keepdim=True) - 8
#   out1 = tmp_2.unsqueeze(0).expand(3, -1, -1)  (free view, same as original)
# ─────────────────────────────────────────────────────────────────────────────

def pattern(tmp_2):
    tmp_5 = tmp_2.unsqueeze(0)
    tmp_6 = tmp_5.expand(3, -1, -1)
    tmp_7 = tmp_6.to(device(type='cuda', index=0))
    max_1 = tmp_7.max(0, keepdim=False)
    tmp_9 = max_1[0]
    max_2 = tmp_9.max(-1, keepdim=True)
    tmp_11 = max_2[0]
    tmp_12 = tmp_11 + 1
    tmp_13 = tmp_12 - 9
    return (tmp_13, tmp_7)


def replacement_args(tmp_2):
    return (tmp_2,)


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel: fused row-max + subtract-8 (available for future use /
# larger batch sizes where it outperforms torch.amax).
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def row_max_kernel(
    tmp2_ptr, out0_ptr,
    B, S,
    BLOCK_S: tl.constexpr,
):
    row     = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_S)
    mask    = offsets < S
    tmp2    = tl.load(tmp2_ptr + row * S + offsets, mask=mask, other=0)
    neg_inf = tl.full([BLOCK_S], -100000, dtype=tl.int64)
    row_max = tl.max(tl.where(mask, tmp2, neg_inf), axis=0)
    tl.store(out0_ptr + row, row_max - 8)


# ─────────────────────────────────────────────────────────────────────────────
# Replacement function — NOT wrapped so torch.compile can see amax+sub and
# fuse them with the surrounding cumsum+sub from the un-replaced graph nodes.
#
# copied_returning_nodes = [out0_node, out1_node]  (len=2)
# match.returning_nodes  = [tmp_13_node, tmp_7_node] (len=2)  ✓
# ─────────────────────────────────────────────────────────────────────────────

def expand_max(tmp_2):
    out0 = tmp_2.amax(dim=-1, keepdim=True) - 8   # [B, 1]
    out1 = tmp_2.unsqueeze(0).expand(3, -1, -1)   # [3,B,S] free view
    return out0, out1


def replacement_func():
    return expand_max