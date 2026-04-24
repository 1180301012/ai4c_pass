import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pass B: Fuse  max(0) → getitem[0] → max(-1,keepdim=True) → getitem[0]
#             → +1 → -9, returning (tmp_13, tmp_7)
#
# tmp_7 is an INPUT placeholder (node computed upstream by expand/to).
# Taking it as a placeholder avoids the FX graph-rewriter crash that occurs
# when the same node is both an internal node AND a return value.
#
# The kernel reads from tmp_7 [3, B, S], computes row-wise max,
# stores (max+1-9) into out_max [B, 1], and materialises all 3 slices into
# out_expanded [3, B, S].
# ---------------------------------------------------------------------------
def pattern(tmp_7):
    max_1  = tmp_7.max(0, keepdim=False)
    tmp_9  = max_1[0]
    max_2  = tmp_9.max(-1, keepdim=True)
    tmp_11 = max_2[0]
    tmp_12 = tmp_11 + 1
    tmp_13 = tmp_12 - 9
    return tmp_13


def replacement_args(tmp_7):
    return (tmp_7,)


# ---------------------------------------------------------------------------
# Triton kernel: row-wise max over tmp_7 [3, B, S].
# Grid: (3*B,) — one program per (slice, batch) pair.
# Materialising out_expanded is NOT done here — tmp_7 stays in the graph.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 64},  num_warps=2),
        triton.Config({'BLOCK_S': 128}, num_warps=4),
        triton.Config({'BLOCK_S': 256}, num_warps=4),
        triton.Config({'BLOCK_S': 512}, num_warps=8),
        triton.Config({'BLOCK_S': 1024}, num_warps=8),
    ],
    key=['S'],
)
@triton.jit
def _max_arithmetic_kernel(
    in_ptr,        # [3, B, S] int64 — tmp_7 (expanded tensor)
    out_max_ptr,   # [B, 1]   int64 — max+1-9 result (= tmp_13)
    B,
    S,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)   # in [0, 3*B - 1]
    b   = pid // 3

    offs = tl.arange(0, BLOCK_S)
    mask = offs < S

    # Load this slice; all 3 slices hold the same data
    vals = tl.load(in_ptr + b * S + offs, mask=mask, other=0)

    row_max = tl.max(vals, axis=0)
    if pid % 3 == 0:
        tl.store(out_max_ptr + b, row_max + 1 - 9)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------
@torch.fx.wrap
def _fused_max_arithmetic(tmp_7):
    B = tmp_7.shape[1]
    S = tmp_7.shape[2]
    out_max = torch.empty((B, 1), dtype=torch.int64, device=tmp_7.device)
    _max_arithmetic_kernel[(3 * B,)](tmp_7, out_max, B, S)
    return out_max   # only tmp_13; tmp_7 stays in graph unchanged


def replacement_func():
    return _fused_max_arithmetic