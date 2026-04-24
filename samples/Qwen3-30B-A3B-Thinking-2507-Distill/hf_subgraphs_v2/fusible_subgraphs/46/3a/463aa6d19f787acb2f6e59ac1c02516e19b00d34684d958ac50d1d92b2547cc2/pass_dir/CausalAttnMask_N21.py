"""
Pass: CausalAttnMask_N21
Matches the final 4 ops of the bfloat16 OPTForSequenceClassification subgraph
(1x1x21x21 causal attention mask finalisation), then replaces with a Triton kernel.
"""

import torch
import triton
import triton.language as tl
from pass_dir.causal_mask_kernel import causal_attention_mask


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 16}),
        triton.Config({'BLOCK_N': 32}),
        triton.Config({'BLOCK_N': 64}),
    ],
    key=['N'],
)
@triton.jit
def _finalise_mask_kernel(
    x_ptr,
    out_ptr,
    N,
    BLOCK_N: tl.constexpr,
):
    """
    Apply the finalisation step to the causal attention mask:
      - row is fully-masked (all in-bounds elements == -3.4028234663852886e+38) → zero the row
      - otherwise keep the row as-is
    Grid: (N,) — one program per query row.
    """
    q = tl.program_id(0)
    k = tl.arange(0, BLOCK_N)
    valid = k < N
    # Load x; out-of-bounds positions get a non-inf value so they don't affect the count
    x = tl.load(x_ptr + q * N + k, mask=valid, other=0.0)
    # Count in-bounds positions where x == -3.4e38
    # Use valid mask to exclude out-of-bounds positions from the count
    count_inf = tl.sum(
        tl.where(valid & (x == -3.4028234663852886e+38), 1, 0),
        axis=0,
    )
    # If all in-bounds positions are masked, zero out the entire row
    out = tl.where(count_inf == BLOCK_N, 0.0, x)
    tl.store(out_ptr + q * N + k, out.to(tl.float32), mask=valid)


@torch.fx.wrap
def finalise_causal_mask(tmp_10):
    """
    Fused replacement for: eq(-3.4e38) → all(-1, keepdim) → ~  → mul
    tmp_10: float32 tensor of shape (1, 1, N, N)
    """
    N = tmp_10.shape[-1]
    out = torch.empty_like(tmp_10)
    _finalise_mask_kernel[(N,)](tmp_10, out, N)
    return out


def pattern(tmp_10):
    tmp_19 = tmp_10.__getattr__('__eq__')(-3.4028234663852886e+38)
    tmp_20 = torch.all(tmp_19, dim=-1, keepdim=True)
    tmp_21 = ~tmp_20
    tmp_22 = tmp_10.mul(tmp_21)
    return tmp_22


def replacement_args(tmp_10):
    return (tmp_10,)


def replacement_func():
    return finalise_causal_mask