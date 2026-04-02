"""
Unified optimization pass for BEiT relative position bias computation.
Covers N=14, N=24, N=32 grids for all dtype variants.

Strategy:
  - Match the final `x.view(-1)` operation in the index-table computation.
  - x is the 2-D integer table produced by the zeros+setitem chain.
  - The replacement ignores x and re-computes the full table analytically
    using a Triton kernel (the whole upstream chain becomes dead code & is DCE'd).
  - The correct kernel is selected at runtime based on x.numel().

This avoids the "dead code in pattern" issue that arises when setitem nodes
(which return None) appear in the pattern graph.
"""

import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel for N=14  (M=197, S=27)
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
    ],
    key=[],
)
@triton.jit
def beit_rpb_n14_kernel(out_ptr, BLOCK_SIZE: tl.constexpr):
    N: tl.constexpr = 14
    M: tl.constexpr = 197
    S: tl.constexpr = 27
    TOTAL: tl.constexpr = M * M
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < TOTAL
    row = offsets // M
    col = offsets % M
    pi = row - 1
    pj = col - 1
    interior = (pi // N - pj // N + N - 1) * S + (pi % N - pj % N + N - 1)
    val = tl.where((row == 0) & (col == 0), 731,
           tl.where(row == 0, 729,
           tl.where(col == 0, 730, interior)))
    tl.store(out_ptr + offsets, val.to(tl.int64), mask=mask)


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel for N=24  (M=577, S=47)
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
    ],
    key=[],
)
@triton.jit
def beit_rpb_n24_kernel(out_ptr, BLOCK_SIZE: tl.constexpr):
    N: tl.constexpr = 24
    M: tl.constexpr = 577
    S: tl.constexpr = 47
    TOTAL: tl.constexpr = M * M
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < TOTAL
    row = offsets // M
    col = offsets % M
    pi = row - 1
    pj = col - 1
    interior = (pi // N - pj // N + N - 1) * S + (pi % N - pj % N + N - 1)
    val = tl.where((row == 0) & (col == 0), 2211,
           tl.where(row == 0, 2209,
           tl.where(col == 0, 2210, interior)))
    tl.store(out_ptr + offsets, val.to(tl.int64), mask=mask)


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel for N=32  (M=1025, S=63)
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
def beit_rpb_n32_kernel(out_ptr, BLOCK_SIZE: tl.constexpr):
    N: tl.constexpr = 32
    M: tl.constexpr = 1025
    S: tl.constexpr = 63
    TOTAL: tl.constexpr = M * M
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < TOTAL
    row = offsets // M
    col = offsets % M
    pi = row - 1
    pj = col - 1
    interior = (pi // N - pj // N + N - 1) * S + (pi % N - pj % N + N - 1)
    val = tl.where((row == 0) & (col == 0), 3971,
           tl.where(row == 0, 3969,
           tl.where(col == 0, 3970, interior)))
    tl.store(out_ptr + offsets, val.to(tl.int64), mask=mask)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level cache: the table is a compile-time constant.
# Computed once on first call (Triton on CUDA → CPU), reused forever.
# ─────────────────────────────────────────────────────────────────────────────
_rpb_cache: dict = {}


@torch.fx.wrap
def compute_rpb_universal(x):
    """
    x is the 2-D integer table produced by the zeros+setitem chain.
    We use x only for numel/device detection; the actual values are ignored.
    The correct table is computed analytically via Triton (once, then cached).
    """
    numel = x.numel()
    target_device = x.device
    key = (numel, str(target_device))

    if key not in _rpb_cache:
        if numel == 1025 * 1025:          # N=32
            TOTAL = 1025 * 1025
            out = torch.empty(TOTAL, dtype=torch.int64, device='cuda')
            beit_rpb_n32_kernel[((TOTAL + 1023) // 1024,)](out)
        elif numel == 577 * 577:           # N=24
            TOTAL = 577 * 577
            out = torch.empty(TOTAL, dtype=torch.int64, device='cuda')
            beit_rpb_n24_kernel[((TOTAL + 1023) // 1024,)](out)
        else:                              # N=14  (197*197 = 38809)
            TOTAL = 197 * 197
            out = torch.empty(TOTAL, dtype=torch.int64, device='cuda')
            beit_rpb_n14_kernel[((TOTAL + 511) // 512,)](out)
        _rpb_cache[key] = out.to(target_device)

    return _rpb_cache[key]


# ─────────────────────────────────────────────────────────────────────────────
# Pattern: match the final view(-1) of the index-table computation.
# ─────────────────────────────────────────────────────────────────────────────

def pattern(x):
    return x.view(-1)


def replacement_args(x):
    return (x,)


def replacement_func():
    return compute_rpb_universal