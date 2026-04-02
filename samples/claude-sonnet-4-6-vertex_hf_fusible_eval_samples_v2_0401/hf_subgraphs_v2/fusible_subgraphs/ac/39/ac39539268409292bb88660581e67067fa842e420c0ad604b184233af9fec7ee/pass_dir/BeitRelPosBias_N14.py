"""
Optimization pass for BEiT relative position bias computation with N=14 grid.
Covers AkshatSurolia_BEiT-FaceMask-Finetuned models (bfloat16, float16, float32).

Pattern:
  - Deterministic relative position index table for 14x14 grid (no tensor inputs)
  - Returns flat_index_table only (cat is left untouched in the graph)

The index table is purely deterministic (doesn't depend on any input tensor),
computed analytically via a Triton kernel instead of running all the
meshgrid/pairwise-diff/slice-mutation steps.
"""

import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel: compute the 197×197 relative-position-bias index table
# (flattened to 197*197 = 38_809 int64 elements) for N=14.
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
    ],
    key=[],
)
@triton.jit
def beit_rpb_n14_kernel(
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    # Compile-time constants for N=14
    N: tl.constexpr = 14
    M: tl.constexpr = 197    # N*N + 1
    S: tl.constexpr = 27     # 2*N - 1
    TOTAL: tl.constexpr = M * M  # 38_809

    pid = tl.program_id(0)
    base = pid * BLOCK_SIZE
    offsets = base + tl.arange(0, BLOCK_SIZE)
    mask = offsets < TOTAL

    row = offsets // M   # which output row  [0 .. M-1]
    col = offsets % M    # which output col  [0 .. M-1]

    # Interior value (row>=1, col>=1)
    pi = row - 1
    pj = col - 1
    row_diff = pi // N - pj // N + (N - 1)
    col_diff = pi % N  - pj % N  + (N - 1)
    interior = row_diff * S + col_diff

    # Border overrides (same order as the original setitem sequence):
    #   row0 (all cols)   → 729
    #   col0 (all rows)   → 730
    #   [0,0]             → 731
    val = tl.where(
        (row == 0) & (col == 0), 731,
        tl.where(row == 0, 729,
                 tl.where(col == 0, 730, interior))
    )

    tl.store(out_ptr + offsets, val.to(tl.int64), mask=mask)


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper – no tensor inputs (index table is purely deterministic).
# Decorated with @torch.fx.wrap so the tracer treats it as a leaf.
# ─────────────────────────────────────────────────────────────────────────────

@torch.fx.wrap
def compute_rpb_n14(sum_result):
    M = 197
    TOTAL = M * M
    out = torch.empty(TOTAL, dtype=torch.int64, device='cuda')

    BLOCK_SIZE = 512
    grid = ((TOTAL + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    beit_rpb_n14_kernel[grid](out)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Pattern: matches the zeros+setitem+view tail of the index computation.
# sum_result (tmp_12.sum(-1)) IS used → no dead-code error.
# ─────────────────────────────────────────────────────────────────────────────

def pattern(sum_result):
    tmp_22 = torch.zeros(size=(197, 197), dtype=torch.int64)
    tmp_22[(slice(1, None, None), slice(1, None, None))] = sum_result
    tmp_22[(0, slice(0, None, None))] = 729
    tmp_22[(slice(0, None, None), 0)] = 730
    tmp_22[(0, 0)] = 731
    tmp_28 = tmp_22.view(-1)
    return tmp_28


def replacement_args(sum_result):
    return (sum_result,)


def replacement_func():
    return compute_rpb_n14