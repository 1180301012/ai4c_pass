import torch
import triton
import triton.language as tl
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Triton kernel: generic cat along dim-0 for any float dtype (N=24 variant)
# ──────────────────────────────────────────────────────────────────────────────
@triton.jit
def _cat_dim0_kernel_n24(
    in1_ptr, in2_ptr, out_ptr,
    n1, n2,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    total = n1 + n2
    mask = offsets < total
    in1_mask = (offsets < n1) & mask
    in2_mask = (~(offsets < n1)) & mask
    val1 = tl.load(in1_ptr + offsets,        mask=in1_mask, other=0.0)
    val2 = tl.load(in2_ptr + (offsets - n1), mask=in2_mask, other=0.0)
    out = tl.where(in1_mask, val1, val2)
    tl.store(out_ptr + offsets, out, mask=mask)


# ──────────────────────────────────────────────────────────────────────────────
# Pattern: matches ONLY torch.cat([in_1, in_0]) for N=24 graphs.
# ──────────────────────────────────────────────────────────────────────────────
def pattern(in_0, in_1):
    tmp_0 = torch.cat([in_1, in_0])
    return tmp_0


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ──────────────────────────────────────────────────────────────────────────────
# Replacement: Triton cat kernel
# ──────────────────────────────────────────────────────────────────────────────
@torch.fx.wrap
def _beit_n24_cat(in_0, in_1):
    n1 = in_1.numel()
    n2 = in_0.numel()
    total_rows = in_1.shape[0] + in_0.shape[0]
    width = in_1.shape[1]
    out_cat = torch.empty((total_rows, width), dtype=in_1.dtype, device=in_1.device)

    BLOCK_SIZE = 1024
    grid = ((n1 + n2 + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _cat_dim0_kernel_n24[grid](in_1, in_0, out_cat, n1, n2, BLOCK_SIZE=BLOCK_SIZE)
    return out_cat


def replacement_func():
    return _beit_n24_cat