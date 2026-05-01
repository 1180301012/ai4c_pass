import torch
import triton
import triton.language as tl
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Triton kernel: generic dim-0 cat (handles any float dtype).
# Two separate masked loads → no tl.where overhead, each block touches exactly
# the memory it needs.
# BLOCK_SIZE=1024, num_warps=8 → 8 warps × 32 = 256 threads/block,
# each thread processes 4 elements – good for hiding memory latency.
# ──────────────────────────────────────────────────────────────────────────────
@triton.jit
def _cat_k1024(in1_ptr, in2_ptr, out_ptr, n1, n2, BLOCK_SIZE: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = n1 + n2
    mask  = offs < total
    m1 = (offs < n1) & mask
    m2 = (~(offs < n1)) & mask

    # Load each source separately – masked loads avoid unnecessary bandwidth
    v1 = tl.load(in1_ptr + offs,        mask=m1, other=0.0)
    v2 = tl.load(in2_ptr + (offs - n1), mask=m2, other=0.0)

    # Merge: for any position only one of m1/m2 is True
    out = tl.where(m1, v1, v2)
    tl.store(out_ptr + offs, out, mask=mask)


# ──────────────────────────────────────────────────────────────────────────────
# Pattern: matches torch.cat([in_1, in_0]) for any BEiT model variant.
# ──────────────────────────────────────────────────────────────────────────────
def pattern(in_0, in_1):
    tmp_0 = torch.cat([in_1, in_0])
    return tmp_0


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ──────────────────────────────────────────────────────────────────────────────
# Replacement: Triton dim-0 cat with num_warps=8
# ──────────────────────────────────────────────────────────────────────────────
@torch.fx.wrap
def _beit_cat(in_0, in_1):
    n1 = in_1.numel()
    n2 = in_0.numel()
    out_cat = torch.empty(
        (in_1.shape[0] + in_0.shape[0], in_1.shape[1]),
        dtype=in_1.dtype, device=in_1.device)
    BS = 1024
    grid = ((n1 + n2 + BS - 1) // BS,)
    _cat_k1024[grid](in_1, in_0, out_cat, n1, n2, BLOCK_SIZE=BS, num_warps=8)
    return out_cat


def replacement_func():
    return _beit_cat