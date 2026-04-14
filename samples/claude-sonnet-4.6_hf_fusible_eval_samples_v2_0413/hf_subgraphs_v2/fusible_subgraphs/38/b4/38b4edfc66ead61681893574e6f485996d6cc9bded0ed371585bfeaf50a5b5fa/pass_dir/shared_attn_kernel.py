"""
Shared Triton kernel and dispatch wrapper for attention-mask fusion passes.

Kernel design: 1 CTA, BLOCK_COL threads per row, loop over all N rows.
  • in_0 loaded ONCE into registers; reused across all N iterations
  • Single kernel launch (no separate torch.full needed)
  • Loop unrolled at compile-time (N is tl.constexpr)
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Single-block kernel: loops over all N rows, loads in_0 once into registers.
# Writes both 0.0 (causal + real) and NEG_INF (otherwise) for every position.
# ---------------------------------------------------------------------------
@triton.jit
def _attn_mask_kernel(
    in_ptr,          # [1, N] int64 – attention mask
    out_ptr,         # [1, 1, N, N] float32 output
    N: tl.constexpr,
    BLOCK_COL: tl.constexpr,  # power-of-2 >= N
):
    col      = tl.arange(0, BLOCK_COL)
    active   = col < N

    # Load all N attention-mask values ONCE (stay in registers for all rows)
    attn_val = tl.load(in_ptr + col, mask=active, other=0)

    NEG_INF  = -3.4028234663852886e+38

    # Unrolled over all N rows (N is constexpr)
    for row in range(N):
        causal   = col  <= row
        is_real  = attn_val == 1
        val      = tl.where(causal & is_real, 0.0, NEG_INF)
        tl.store(out_ptr + row * N + col, val.to(tl.float32), mask=active)


# ---------------------------------------------------------------------------
# Shared dispatch wrapper
# ---------------------------------------------------------------------------
@torch.fx.wrap
def attn_mask_fusion(in_0, route):
    if route == "n9":
        N         = 9
        BLOCK_COL = 16   # power-of-2 >= 9
    else:               # "n13"
        N         = 13
        BLOCK_COL = 16   # power-of-2 >= 13

    out = torch.empty((1, 1, N, N), dtype=torch.float32, device=in_0.device)
    # Single block launch – 1 CTA handles all N rows via a constexpr loop
    _attn_mask_kernel[(1,)](in_0, out, N=N, BLOCK_COL=BLOCK_COL)
    return out