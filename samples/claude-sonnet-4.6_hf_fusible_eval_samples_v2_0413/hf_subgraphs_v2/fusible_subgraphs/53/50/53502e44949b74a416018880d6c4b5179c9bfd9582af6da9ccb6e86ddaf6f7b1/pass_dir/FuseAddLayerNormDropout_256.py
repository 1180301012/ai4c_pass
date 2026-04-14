import torch
import triton
import triton.language as tl


# ─── Pattern: tok_emb·16 + pos_emb + layer_norm  (4 ops fused) ────────────────
# b = positional-embedding output (result of the arange→embed chain, external).
# Fuses: embedding(tok), *16, add(pos_emb), layer_norm → 1 Triton kernel.
def pattern(in_4, in_1, b, in_3, in_2):
    tok_emb  = torch.nn.functional.embedding(in_4, in_1, 1, None, 2.0, False, False)
    scaled   = tok_emb * 16.0
    combined = scaled + b
    out      = torch.nn.functional.layer_norm(combined, (256,), in_3, in_2, 1e-05)
    return out


def replacement_args(in_4, in_1, b, in_3, in_2):
    return (in_4, in_1, b, in_3, in_2)


# ─── Triton kernel ────────────────────────────────────────────────────────────
@triton.jit
def fused_4op_kernel(
    in4_ptr,      # int64   [N]          flat token IDs
    in1_ptr,      # bf16/f16 [vocab, H]  token embedding table
    b_ptr,        # bf16/f16 [N, H]      positional embedding output
    w_ptr,        # bf16/f16 [H]         LN weight
    bias_ptr,     # bf16/f16 [H]         LN bias
    out_ptr,      # bf16/f16 [N, H]      output
    HIDDEN:  tl.constexpr,   # 256
    PAD_IDX: tl.constexpr,   # 1
    SCALE:   tl.constexpr,   # 16.0
):
    row  = tl.program_id(0)
    offs = tl.arange(0, HIDDEN)
    base = row * HIDDEN

    # ── token embedding (with padding) ───────────────────────────────────────
    token_id = tl.load(in4_ptr  + row)
    tok_emb  = tl.load(in1_ptr  + token_id * HIDDEN + offs).to(tl.float32)
    not_pad  = (token_id != PAD_IDX).to(tl.float32)   # 0.0 if padding, 1.0 otherwise
    tok_emb  = tok_emb * not_pad

    # ── scale + add positional embedding ─────────────────────────────────────
    pos = tl.load(b_ptr + base + offs).to(tl.float32)
    x   = tok_emb * SCALE + pos

    # ── layer norm ────────────────────────────────────────────────────────────
    mean  = tl.sum(x, axis=0) / HIDDEN
    diff  = x - mean
    var   = tl.sum(diff * diff, axis=0) / HIDDEN
    rstd  = 1.0 / tl.sqrt(var + 1e-5)
    x_hat = diff * rstd

    w  = tl.load(w_ptr    + offs).to(tl.float32)
    bv = tl.load(bias_ptr + offs).to(tl.float32)
    y  = x_hat * w + bv

    # auto-casts float32 → output element type (bf16 or f16)
    tl.store(out_ptr + base + offs, y)


# ─── Python wrapper ───────────────────────────────────────────────────────────
@torch.fx.wrap
def fused_4op(in_4, in_1, b, in_3, in_2):
    """
    in_4 : int64    [B, S]       token IDs
    in_1 : bf16/f16 [vocab, 256] token embedding weight
    b    : bf16/f16 [B, S, 256]  positional embedding output (shape reference)
    in_3 : bf16/f16 [256]        LN weight
    in_2 : bf16/f16 [256]        LN bias
    """
    out = torch.empty_like(b)
    fused_4op_kernel[(in_4.numel(),)](
        in_4, in_1, b, in_3, in_2, out,
        HIDDEN=256,
        PAD_IDX=1,
        SCALE=16.0,
    )
    return out


def replacement_func():
    return fused_4op