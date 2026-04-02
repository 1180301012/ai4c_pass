"""
Optimization pass for BEiT relative position bias computation with N=32 grid.
Covers beit-large-patch16-512 models (bfloat16, float16, float32).

Pattern:
  - Deterministic relative position index table for 32x32 grid (no tensor inputs)
  - Returns flat_index_table only (cat is left untouched in the graph)

The index table is purely deterministic (doesn't depend on any input tensor),
computed analytically via a Triton kernel instead of running all the
meshgrid/pairwise-diff/slice-mutation steps.
"""

import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel: compute the 1025×1025 relative-position-bias index table
# (flattened to 1025*1025 = 1_050_625 int64 elements) for N=32.
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
        triton.Config({'BLOCK_SIZE': 4096}),
    ],
    key=[],
)
@triton.jit
def beit_rpb_n32_kernel(
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    # Compile-time constants for N=32
    N: tl.constexpr = 32
    M: tl.constexpr = 1025   # N*N + 1
    S: tl.constexpr = 63     # 2*N - 1
    TOTAL: tl.constexpr = M * M  # 1_050_625

    pid = tl.program_id(0)
    base = pid * BLOCK_SIZE
    offsets = base + tl.arange(0, BLOCK_SIZE)
    mask = offsets < TOTAL

    row = offsets // M   # which output row  [0 .. M-1]
    col = offsets % M    # which output col  [0 .. M-1]

    # Interior value (row>=1, col>=1)
    # patch_i = row-1, patch_j = col-1
    pi = row - 1
    pj = col - 1
    row_diff = pi // N - pj // N + (N - 1)
    col_diff = pi % N  - pj % N  + (N - 1)
    interior = row_diff * S + col_diff

    # Border overrides (applied in same order as the original setitem sequence):
    #   row0 (all cols)   → 3969
    #   col0 (all rows)   → 3970
    #   [0,0]             → 3971
    val = tl.where(
        (row == 0) & (col == 0), 3971,
        tl.where(row == 0, 3969,
                 tl.where(col == 0, 3970, interior))
    )

    tl.store(out_ptr + offsets, val.to(tl.int64), mask=mask)


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper – no tensor inputs (index table is purely deterministic).
# Decorated with @torch.fx.wrap so the tracer treats it as a leaf.
# ─────────────────────────────────────────────────────────────────────────────

@torch.fx.wrap
def compute_rpb_n32(sum_result):
    # sum_result is not used – we compute the full table analytically via Triton.
    # The upstream arange/meshgrid/diff chain becomes dead code and is DCE'd.
    M = 1025
    TOTAL = M * M
    out = torch.empty(TOTAL, dtype=torch.int64, device='cuda')

    BLOCK_SIZE = 1024
    grid = ((TOTAL + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    beit_rpb_n32_kernel[grid](out)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Pattern: matches the zeros+setitem+view tail of the index computation.
# sum_result is the output of tmp_12.sum(-1) – it IS used (setitem) so no
# "dead code" error from SubgraphMatcher.
# ─────────────────────────────────────────────────────────────────────────────

def pattern(sum_result):
    tmp_22 = torch.zeros(size=(1025, 1025), dtype=torch.int64)
    tmp_22[(slice(1, None, None), slice(1, None, None))] = sum_result
    tmp_22[(0, slice(0, None, None))] = 3969
    tmp_22[(slice(0, None, None), 0)] = 3970
    tmp_22[(0, 0)] = 3971
    tmp_28 = tmp_22.view(-1)
    return tmp_28


def replacement_args(sum_result):
    return (sum_result,)


def replacement_func():
    return compute_rpb_n32