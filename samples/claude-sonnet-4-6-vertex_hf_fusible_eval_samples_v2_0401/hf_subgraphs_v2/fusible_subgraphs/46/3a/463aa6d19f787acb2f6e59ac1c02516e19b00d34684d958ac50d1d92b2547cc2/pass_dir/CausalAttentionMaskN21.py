"""
Pass: CausalAttentionMaskN21
Fuses the full causal + padding attention mask computation for N=21 into one Triton kernel.

Pattern strategy:
  - The setitem `tmp_10[...] = tmp_17` is an in-place side-effect with NO users in the
    FX graph (eq and mul both use the original clone node `tmp_10`, not the setitem output).
  - Including setitem in the pattern causes "dead code" in the SubgraphMatcher.
  - Solution: skip the setitem in the pattern.
  - But then masked_fill (tmp_17) and clone (tmp_10) are used OUTSIDE the matched subgraph
    (by the unmatched setitem), so per the guidelines they MUST appear in the pattern return.
  - Returning (tmp_22, tmp_17, tmp_10) makes all nodes live and exposes the right outputs.
  - After replacement, setitem's inputs are the replacement's outputs for tmp_17 and tmp_10.
    Since setitem has no users, DCE removes it automatically.
"""
import torch
import triton
import triton.language as tl
from torch import device


# ---------------------------------------------------------------------------
# Triton kernel – computes all three outputs in one pass
# ---------------------------------------------------------------------------
@triton.jit
def causal_attn_mask_kernel(
    in0_ptr,        # [B, N]  int64
    out_final_ptr,  # [B, 1, N, N]  float32  – tmp_22 (with row zeroing)
    out_inter_ptr,  # [B, 1, N, N]  float32  – tmp_17 (without row zeroing)
    out_causal_ptr, # [B, 1, N, N]  float32  – tmp_10 (causal mask only)
    B,
    N,
    BLOCK_N: tl.constexpr,
):
    """Grid: (B * N,) – each program handles one (batch b, row i)."""
    pid = tl.program_id(0)
    b = pid // N
    i = pid % N

    NEG_INF = -3.4028234663852886e+38

    j = tl.arange(0, BLOCK_N)
    col_ok = j < N

    # Load attention mask
    in0 = tl.load(in0_ptr + b * N + j, mask=col_ok, other=1)

    # ── Causal mask (tmp_10): -inf for j > i, else 0 ──────────────────────
    causal_masked = j > i
    causal_val = tl.where(causal_masked & col_ok, NEG_INF, 0.0)
    tl.store(out_causal_ptr + b * N * N + i * N + j, causal_val, mask=col_ok)

    # ── Intermediate mask (tmp_17): -inf for masked, else 0 ───────────────
    pad_masked = in0 == 0
    is_masked = causal_masked | pad_masked
    inter_val = tl.where(is_masked & col_ok, NEG_INF, 0.0)
    tl.store(out_inter_ptr + b * N * N + i * N + j, inter_val, mask=col_ok)

    # ── Row-zeroed final mask (tmp_22) ─────────────────────────────────────
    n_valid = tl.sum(((~is_masked) & col_ok).to(tl.int32))
    scale = tl.where(n_valid > 0, 1.0, 0.0)
    final_val = inter_val * scale
    tl.store(out_final_ptr + b * N * N + i * N + j, final_val, mask=col_ok)


@torch.fx.wrap
def causal_attn_mask_21(in_0):
    B = in_0.shape[0]
    N = in_0.shape[1]

    out_final  = torch.empty(B, 1, N, N, dtype=torch.float32, device=in_0.device)
    out_inter  = torch.empty(B, 1, N, N, dtype=torch.float32, device=in_0.device)
    out_causal = torch.empty(B, 1, N, N, dtype=torch.float32, device=in_0.device)

    causal_attn_mask_kernel[(B * N,)](
        in_0,
        out_final, out_inter, out_causal,
        B, N,
        BLOCK_N=32,
    )
    # Return order matches pattern: (tmp_22, tmp_17, tmp_10)
    return out_final, out_inter, out_causal


# ---------------------------------------------------------------------------
# Pattern – skips the in-place setitem (dead code in FX graph).
# Returns (tmp_22, tmp_17, tmp_10) so every internal node has a user.
# ---------------------------------------------------------------------------
def pattern(in_0):
    tmp_1 = torch.arange(0, 21, device=device(type='cuda', index=0))
    tmp_2 = torch.full((21, 21), fill_value=-3.4028234663852886e+38, dtype=torch.float32, device=device(type='cuda', index=0))
    tmp_3 = torch.triu(tmp_2, diagonal=1)
    tmp_4 = torch.arange(21, device=device(type='cuda', index=0))
    tmp_5 = tmp_1.reshape(-1, 1)
    tmp_6 = tmp_4 > tmp_5
    tmp_3 *= tmp_6
    tmp_7 = tmp_3
    tmp_8 = tmp_7[(None, None, slice(None, None, None), slice(None, None, None))]
    tmp_9 = tmp_8.expand(1, 1, -1, -1)
    tmp_10 = tmp_9.clone()
    tmp_11 = tmp_10[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 21, None))]
    tmp_12 = in_0[(slice(None, None, None), None, None, slice(None, None, None))]
    tmp_13 = tmp_12.to(device(type='cuda', index=0))
    tmp_14 = tmp_11 + tmp_13
    tmp_15 = tmp_14.__eq__(0)
    tmp_16 = tmp_10[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 21, None))]
    tmp_17 = tmp_16.masked_fill(tmp_15, -3.4028234663852886e+38)
    # NOTE: setitem is intentionally omitted (it is a dead-code side-effect in the FX graph).
    # tmp_10 (clone) is used directly by eq and mul – matching the model's graph structure.
    tmp_19 = tmp_10.__eq__(-3.4028234663852886e+38)
    tmp_20 = torch.all(tmp_19, dim=-1, keepdim=True)
    tmp_21 = ~tmp_20
    tmp_22 = tmp_10.mul(tmp_21)
    # Return all three observable values:
    #   tmp_22 – used by model's return
    #   tmp_17 – used by model's setitem (outside matched subgraph)
    #   tmp_10 – used by model's setitem (outside matched subgraph)
    return tmp_22, tmp_17, tmp_10


def replacement_args(in_0):
    return (in_0,)


def replacement_func():
    return causal_attn_mask_21