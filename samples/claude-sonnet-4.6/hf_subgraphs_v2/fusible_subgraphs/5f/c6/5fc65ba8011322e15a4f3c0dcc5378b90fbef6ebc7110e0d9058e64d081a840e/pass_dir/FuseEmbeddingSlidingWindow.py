"""
Fuse: embedding lookup + sliding-window concat into a single Triton kernel.

Original computation:
  tmp_2 = F.embedding(in_0, in_1, 0, None, 2.0, False, False)  # [B, S, D]
  tmp_3 = tmp_2[:, 1:]                                           # next tokens
  tmp_4 = F.pad(tmp_3, [0,0,0,1,0,0], 'constant', 0.0)         # [B, S, D], zero at end
  tmp_5 = tmp_2[:, :-1]                                          # prev tokens
  tmp_6 = F.pad(tmp_5, [0,0,1,0,0,0], 'constant', 0.0)         # [B, S, D], zero at start
  tmp_7 = torch.cat([tmp_4, tmp_2, tmp_6], dim=2)               # [B, S, 3*D]

Fused kernel writes [next_emb | curr_emb | prev_emb] directly to output.
For position (b, s):
  out[b, s, 0:D]    = weight[in_0[b, s+1]]  if s < S-1 else 0
  out[b, s, D:2D]   = weight[in_0[b, s]]
  out[b, s, 2D:3D]  = weight[in_0[b, s-1]]  if s > 0   else 0
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern to match
# ---------------------------------------------------------------------------
def pattern(in_0, in_1):
    tmp_2 = torch.nn.functional.embedding(in_0, in_1, 0, None, 2.0, False, False)
    tmp_3 = tmp_2[(slice(None, None, None), slice(1, None, None))]
    tmp_4 = torch.nn.functional.pad(tmp_3, [0, 0, 0, 1, 0, 0], 'constant', 0.0)
    tmp_5 = tmp_2[(slice(None, None, None), slice(None, -1, None))]
    tmp_6 = torch.nn.functional.pad(tmp_5, [0, 0, 1, 0, 0, 0], 'constant', 0.0)
    tmp_7 = torch.cat([tmp_4, tmp_2, tmp_6], dim=2)
    return tmp_7


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_E': 128}, num_warps=4),
        triton.Config({'BLOCK_E': 128}, num_warps=8),
        triton.Config({'BLOCK_E': 256}, num_warps=4),
        triton.Config({'BLOCK_E': 256}, num_warps=8),
    ],
    key=['embed_dim'],
)
@triton.jit
def _fused_emb_sliding_kernel(
    idx_ptr,        # [B*S] int64, flat token indices
    w_ptr,          # [V, D] float, embedding weight (contiguous)
    out_ptr,        # [B*S, 3*D] float, output
    seq_len,        # S (runtime)
    embed_dim,      # D (runtime)
    BLOCK_E: tl.constexpr,  # must be >= embed_dim, power-of-2
):
    # Each program handles one (batch, seq) token position
    pid = tl.program_id(0)
    seq_idx = pid % seq_len

    # Element offsets within the embedding dimension
    e = tl.arange(0, BLOCK_E)
    e_mask = e < embed_dim

    # ---- Load token indices ----
    curr_tok = tl.load(idx_ptr + pid)

    # Boundary flags (scalar tl.int1)
    has_next = seq_idx < (seq_len - 1)
    has_prev = seq_idx > 0

    # Safe neighbor addresses: add/subtract 0 at boundary so pointer is valid
    next_tok = tl.load(idx_ptr + pid + has_next)
    prev_tok = tl.load(idx_ptr + pid - has_prev)

    # ---- Fetch embeddings ----
    curr_emb = tl.load(w_ptr + curr_tok * embed_dim + e,
                       mask=e_mask, other=0.0)
    # Mask with has_next / has_prev so boundary positions get 0.0
    next_emb = tl.load(w_ptr + next_tok * embed_dim + e,
                       mask=e_mask & has_next, other=0.0)
    prev_emb = tl.load(w_ptr + prev_tok * embed_dim + e,
                       mask=e_mask & has_prev, other=0.0)

    # ---- Write output: [next | curr | prev] ----
    out_base = pid * 3 * embed_dim
    tl.store(out_ptr + out_base + e,                    next_emb, mask=e_mask)
    tl.store(out_ptr + out_base + embed_dim + e,        curr_emb, mask=e_mask)
    tl.store(out_ptr + out_base + 2 * embed_dim + e,   prev_emb, mask=e_mask)


# ---------------------------------------------------------------------------
# Python wrapper (must be @torch.fx.wrap so FX doesn't trace inside it)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_emb_sliding(in_0, in_1):
    B = in_0.shape[0]
    S = in_0.shape[1]
    D = in_1.shape[1]
    BLOCK_E = triton.next_power_of_2(D)

    # Output tensor: [B, S, 3*D]
    out = torch.empty((B, S, 3 * D), dtype=in_1.dtype, device=in_1.device)

    # Launch one program per token position
    _fused_emb_sliding_kernel[(B * S,)](
        idx_ptr=in_0,
        w_ptr=in_1,
        out_ptr=out,
        seq_len=S,
        embed_dim=D,
        BLOCK_E=BLOCK_E,
    )

    return out


# ---------------------------------------------------------------------------
# Replacement entry point
# ---------------------------------------------------------------------------
def replacement_func():
    return fused_emb_sliding