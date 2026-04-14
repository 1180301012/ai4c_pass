import torch
import triton
import triton.language as tl
from torch import device as torch_device


# ─── Full 8-op pattern: both embeddings + scale + add + layer_norm ────────────
# Covers: embed(tok)*16 + arange→expand→+2→embed(pos) → add → layer_norm
def pattern(in_4, in_1, in_0, in_3, in_2):
    tok_emb  = torch.nn.functional.embedding(in_4, in_1, 1, None, 2.0, False, False)
    scaled   = tok_emb * 16.0
    arange_val = torch.arange(0, 1, dtype=torch.int64, device=torch_device(type='cuda', index=0))
    expanded   = arange_val.expand(1, -1)
    pos_idx    = expanded + 2
    pos_emb  = torch.nn.functional.embedding(pos_idx, in_0, None, None, 2.0, False, False)
    combined = scaled + pos_emb
    out      = torch.nn.functional.layer_norm(combined, (256,), in_3, in_2, 1e-05)
    return out


def replacement_args(in_4, in_1, in_0, in_3, in_2):
    return (in_4, in_1, in_0, in_3, in_2)


# ─── Triton kernel ────────────────────────────────────────────────────────────
@triton.jit
def fused_full_kernel(
    in4_ptr,       # int64   [N]           flat token IDs
    in1_ptr,       # bf16/f16 [vocab,256]  token embedding table
    in0_ptr,       # bf16/f16 [514,256]    positional embedding table
    w_ptr,         # bf16/f16 [256]         LN weight
    bias_ptr,      # bf16/f16 [256]         LN bias
    out_ptr,       # bf16/f16 [N,256]       output (written row-major)
    HIDDEN:  tl.constexpr,   # 256
    PAD_IDX: tl.constexpr,   # 1  (token embedding padding_idx)
    POS_IDX: tl.constexpr,   # 2  (arange(0,1)+2 is always 2)
    SCALE:   tl.constexpr,   # 16.0
):
    row  = tl.program_id(0)
    offs = tl.arange(0, HIDDEN)
    base = row * HIDDEN

    # ── token embedding ───────────────────────────────────────────────────────
    token_id = tl.load(in4_ptr + row)                    # scalar int64
    tok_emb  = tl.load(in1_ptr + token_id * HIDDEN + offs).to(tl.float32)
    # padding_idx=1: zero out embedding if token_id == PAD_IDX
    not_pad  = (token_id != PAD_IDX).to(tl.float32)      # 0.0 or 1.0
    tok_emb  = tok_emb * not_pad

    # ── positional embedding (position is always POS_IDX = 2) ─────────────────
    pos_emb = tl.load(in0_ptr + POS_IDX * HIDDEN + offs).to(tl.float32)

    # ── fused: scale + add ────────────────────────────────────────────────────
    x = tok_emb * SCALE + pos_emb

    # ── layer norm ────────────────────────────────────────────────────────────
    mean  = tl.sum(x, axis=0) / HIDDEN
    diff  = x - mean
    var   = tl.sum(diff * diff, axis=0) / HIDDEN
    rstd  = 1.0 / tl.sqrt(var + 1e-5)
    x_hat = diff * rstd

    w  = tl.load(w_ptr    + offs).to(tl.float32)
    bv = tl.load(bias_ptr + offs).to(tl.float32)
    y  = x_hat * w + bv

    # store – Triton auto-casts float32 to the pointer element-type
    tl.store(out_ptr + base + offs, y)


# ─── Python wrapper ───────────────────────────────────────────────────────────
@torch.fx.wrap
def fused_full_graph(in_4, in_1, in_0, in_3, in_2):
    """
    in_4 : int64    [batch, seq]       token IDs
    in_1 : bf16/f16 [vocab, 256]       token embedding weight
    in_0 : bf16/f16 [514, 256]         positional embedding weight
    in_3 : bf16/f16 [256]              LN weight
    in_2 : bf16/f16 [256]              LN bias
    """
    HIDDEN = 256
    N  = in_4.numel()          # batch * seq_len (= 1 for target graphs)
    B  = in_4.shape[0]         # batch size
    S  = in_4.shape[1]         # seq length
    out = torch.empty(B, S, HIDDEN, dtype=in_1.dtype, device=in_1.device)

    fused_full_kernel[(N,)](
        in_4,
        in_1,
        in_0,
        in_3,
        in_2,
        out,
        HIDDEN=HIDDEN,
        PAD_IDX=1,
        POS_IDX=2,
        SCALE=16.0,
    )
    return out


def replacement_func():
    return fused_full_graph