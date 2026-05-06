import torch
import triton
import triton.language as tl
from torch import device


# ──────────────────────────────────────────────────────────────────────────────
# Pattern: mirrors the full forward() of the jina-reranker subgraph exactly
# Both tmp_7 (attention-multiplicand [3,B,S]) and tmp_13 (scalar result)
# appear in the model's return, so both must be produced here.
# ──────────────────────────────────────────────────────────────────────────────
def pattern(in_0, in_1):
    tmp_1 = in_1.cumsum(-1)
    tmp_2 = tmp_1 - 1
    tmp_3 = in_0.__eq__(0)
    tmp_4 = tmp_2.masked_fill_(tmp_3, 1)
    tmp_5 = tmp_4.unsqueeze(0)
    tmp_6 = tmp_5.expand(3, -1, -1)
    tmp_7 = tmp_6.to(device(type='cuda', index=0))
    max_1 = tmp_7.max(0, keepdim=False)
    tmp_9 = max_1[0]
    max_2 = tmp_9.max(-1, keepdim=True)
    tmp_11 = max_2[0]
    tmp_12 = tmp_11 + 1
    tmp_13 = tmp_12 - 9
    return (tmp_13, tmp_7)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ──────────────────────────────────────────────────────────────────────────────
# Fused Triton kernel
#  • Grid: (ceil(B*S / BLOCK_ROWS),)
#  • Each program owns BLOCK_ROWS rows of the [B, S] input
#
# Steps per row c:
#   a)  cumsum = cumsum(in_1[c]) - 1   → base attention score
#   b)  if in_0[c] == 0 → score = max(cumsum, 1)
#   c)  store to out_corners[c]  (tmp_7 first batch)
#   d)  scatter to out_3x[c]     (tmp_7 batches 1 & 2)
#   e)  per-row max over out_3x  → max3[c]
#   f)  per-col max over max3   → maxcol[c]
#   g)  result[c] = maxcol[c] - 8  (maps to tmp_13)
# ──────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_ROWS': 64,  'BLOCK_S': 64},  num_warps=2),
        triton.Config({'BLOCK_ROWS': 128, 'BLOCK_S': 64},  num_warps=4),
        triton.Config({'BLOCK_ROWS': 32,  'BLOCK_S': 128}, num_warps=4),
        triton.Config({'BLOCK_ROWS': 64,  'BLOCK_S': 128}, num_warps=4),
        triton.Config({'BLOCK_ROWS': 16,  'BLOCK_S': 256}, num_warps=4),
        triton.Config({'BLOCK_ROWS': 32,  'BLOCK_S': 256}, num_warps=4),
        triton.Config({'BLOCK_ROWS': 64,  'BLOCK_S': 256}, num_warps=8),
        triton.Config({'BLOCK_ROWS': 128, 'BLOCK_S': 256}, num_warps=8),
        triton.Config({'BLOCK_ROWS': 256, 'BLOCK_S': 256}, num_warps=8),
        triton.Config({'BLOCK_ROWS': 512, 'BLOCK_S': 256}, num_warps=8),
        triton.Config({'BLOCK_ROWS': 1024,'BLOCK_S': 256}, num_warps=8),
    ],
    key=['n_rows'],
)
@triton.jit
def _fused_reranker_kernel(
    in_0_ptr,   # [B, S] int64
    in_1_ptr,   # [B, S] int64
    out_corners_ptr,  # [B*S] int64  ← tmp_13 flat
    out_3x_ptr,       # [3*B*S] int64  ← tmp_7 flat (batch-0 in first half)
    n_rows,           # B * S
    BLOCK_ROWS: tl.constexpr,
    BLOCK_S:    tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_ROWS
    row_offsets  = row_start + tl.arange(0, BLOCK_ROWS)
    row_mask     = row_offsets < n_rows

    # ── load in_0 and in_1 rows ──────────────────────────────────────────────
    in_0_row = tl.load(in_0_ptr + row_offsets, mask=row_mask, other=0)
    in_1_row = tl.load(in_1_ptr + row_offsets, mask=row_mask, other=0)
    col_offsets = tl.arange(0, BLOCK_S)

    cumsum = tl.cumsum(in_1_row, axis=0)  # shape [BLOCK_ROWS]

    # ── compute attention scores ─────────────────────────────────────────────
    # cumsum - 1  (then clamp to >= 1 where in_0 was 0)
    base   = cumsum - 1                                 # int64
    is_zero = (in_0_row == 0)

    score  = tl.where(is_zero, tl.maximum(base, 1), base)

    # ── store to tmp_7 batch-0 (first row of out_3x) ─────────────────────────
    tl.store(out_3x_ptr + row_offsets, score, mask=row_mask)

    # ── gather + scatter to batches 1 & 2 of tmp_7 ───────────────────────────
    enc = row_offsets[:, None]        # [BLOCK_ROWS, 1]
    flat = enc * BLOCK_S + col_offsets  # [BLOCK_ROWS, BLOCK_S]
    flat_mask = row_mask[:, None] & (col_offsets[None, :] < BLOCK_S)

    tl.store(
        out_3x_ptr + BLOCK_S + flat,
        score[:, None],
        mask=flat_mask,
    )
    tl.store(
        out_3x_ptr + 2 * BLOCK_S + flat,
        score[:, None],
        mask=flat_mask,
    )

    # ── row-wise max over 3 batches → max3 [BLOCK_ROWS] ─────────────────────
    max3 = tl.maximum(
        tl.maximum(
            tl.load(out_3x_ptr + 0 * BLOCK_S + row_offsets, mask=row_mask, other=0),
            tl.load(out_3x_ptr + 1 * BLOCK_S + row_offsets, mask=row_mask, other=0),
        ),
        tl.load(out_3x_ptr + 2 * BLOCK_S + row_offsets, mask=row_mask, other=0),
    )

    # ── col-wise max (BLOCK_S is >= S, so we can safely reduce this dimension) ─
    maxcol = tl.max(max3, axis=0)   # scalar

    # ── result = maxcol + 1 - 9 ───────────────────────────────────────────────
    result = maxcol - 8

    tl.store(out_corners_ptr + row_offsets, result, mask=row_mask)


@torch.fx.wrap
def fused_reranker(in_0, in_1):
    B = in_0.shape[0]
    S = in_0.shape[1]
    n_rows = B * S

    out_corners = torch.empty((n_rows,), dtype=torch.int64, device=in_0.device)
    # layout: [3, B, S], contiguous → flat size 3*B*S, starting at offset 0 for batch-0
    out_3x = torch.empty((3 * B * S,), dtype=torch.int64, device=in_0.device)

    def grid(meta):
        return ((n_rows + meta['BLOCK_ROWS'] - 1) // meta['BLOCK_ROWS'],)

    _fused_reranker_kernel[grid](
        in_0, in_1,
        out_corners,
        out_3x,
        n_rows,
    )

    tmp_7   = out_3x.view(3, B, S)
    tmp_13  = out_corners.view(B, S)   # scalar reduction along S (B*S / B = S row-per-row)
    return (tmp_13, tmp_7)


def replacement_func():
    return fused_reranker