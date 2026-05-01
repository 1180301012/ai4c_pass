import torch
import triton
import triton.language as tl
from torch import device
from pass_dir.shared_dispatch import _dispatch  # shared object → fixes replacement_func_limit

# ---------------------------------------------------------------------------
# HEAD PASS for N=21: replaces ops 1-10 (causal mask construction) with
# a single Triton kernel.  Pattern has NO external inputs — everything is
# generated from constants.  Output = the initial causal mask [1,1,21,21].
#
# This reduces ~7 kernel launches (arange×2, full, triu, mul, expand, clone)
# to 1, amortising the Triton launch overhead over more work.
# ---------------------------------------------------------------------------

@triton.jit
def _causal_mask_kernel_N21(out_ptr, N: tl.constexpr, BLOCK_N: tl.constexpr):
    """Build the causal mask: out[row, col] = -3.4e38 if col>row else 0."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    col_mask = cols < N
    NEG_INF = -3.4028234663852886e+38
    out_val = tl.where(cols > row,
                       tl.full([BLOCK_N], NEG_INF, dtype=tl.float32),
                       tl.zeros([BLOCK_N], dtype=tl.float32))
    tl.store(out_ptr + row * N + cols, out_val, mask=col_mask)


@torch.fx.wrap
def _build_causal_N21():
    """Replacement for the causal-mask-construction subgraph (N=21)."""
    out = torch.empty((1, 1, 21, 21), dtype=torch.float32, device='cuda:0')
    _causal_mask_kernel_N21[(21,)](out, N=21, BLOCK_N=32)
    return out


# ---------------------------------------------------------------------------
# Pattern: the 10-op causal mask construction for N=21.
# No external inputs (all ops create constants) → no placeholder nodes.
# No dead code: every node is connected to the clone output.
# ---------------------------------------------------------------------------
def pattern():
    tmp_1 = torch.arange(0, 21, device=device(type='cuda', index=0))
    tmp_2 = torch.full((21, 21), fill_value=-3.4028234663852886e+38, dtype=torch.float32, device=device(type='cuda', index=0))
    tmp_3 = torch.triu(tmp_2, diagonal=1)
    tmp_4 = torch.arange(21, device=device(type='cuda', index=0))
    tmp_5 = tmp_1.reshape(-1, 1)
    tmp_6 = tmp_4 > tmp_5
    tmp_7 = tmp_3 * tmp_6
    tmp_8 = tmp_7[(None, None, slice(None, None, None), slice(None, None, None))]
    tmp_9 = tmp_8.expand(1, 1, -1, -1)
    tmp_10 = tmp_9.clone()
    return (tmp_10,)


def replacement_args():
    return ()


def replacement_func():
    return _dispatch