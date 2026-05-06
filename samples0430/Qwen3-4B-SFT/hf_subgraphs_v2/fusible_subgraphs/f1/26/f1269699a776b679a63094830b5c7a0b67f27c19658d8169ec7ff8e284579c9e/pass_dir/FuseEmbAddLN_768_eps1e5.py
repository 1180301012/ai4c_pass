import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────────
# Pattern: two embeddings + add + layer_norm + dropout (inference, no-op)
# ─────────────────────────────────────────────────────────────────────────────
def pattern(in_0, in_4, in_5, in_3, in_2, in_1):
    tmp_5 = torch.nn.functional.embedding(in_0, in_4, 1, None, 2.0, False, False)
    tmp_6 = torch.nn.functional.embedding(in_5, in_3, 1, None, 2.0, False, False)
    tmp_7 = tmp_5 + tmp_6
    tmp_8 = torch.nn.functional.layer_norm(tmp_7, (768,), in_2, in_1, 1e-05)
    tmp_9 = torch.nn.functional.dropout(tmp_8, 0.1, False, False)
    return tmp_9


def replacement_args(in_0, in_4, in_5, in_3, in_2, in_1):
    return (in_0, in_4, in_5, in_3, in_2, in_1)


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel: fused word-embedding + position-embedding gather + LayerNorm
#
# Each program handles one (batch, seq) token position and all H hidden dims.
# We use BLOCK_H (constexpr) ≥ H so the entire hidden vector fits in one block.
# All arithmetic is done in float32 for numerical parity with torch.nn.LayerNorm.
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_H': 1024}, num_warps=4),
        triton.Config({'BLOCK_H': 1024}, num_warps=8),
        triton.Config({'BLOCK_H': 1024}, num_warps=16),
    ],
    key=['H'],
)
@triton.jit
def _fused_emb_ln_kernel(
    word_ids_ptr,   # [batch * seq]  int64 – flat token ids
    pos_ids_ptr,    # [batch * seq]  int64 – flat position ids
    word_emb_ptr,   # [word_vocab, H] float – word embeddings
    pos_emb_ptr,    # [pos_vocab,  H] float – position embeddings
    ln_w_ptr,       # [H]            float – LayerNorm weight
    ln_b_ptr,       # [H]            float – LayerNorm bias
    out_ptr,        # [batch * seq, H] float – output
    H,              # hidden dim (runtime int)
    seq_len,        # sequence length (runtime int)
    BLOCK_H: tl.constexpr,   # tile ≥ H, power-of-2
):
    prog_id = tl.program_id(0)    # one program per (batch, seq) element
    h_range = tl.arange(0, BLOCK_H)  # shape [BLOCK_H]
    mask = h_range < H             # which lanes are in-bounds

    # ---- load token ids ------------------------------------------------
    token_id = tl.load(word_ids_ptr + prog_id)  # scalar int64
    pos_id   = tl.load(pos_ids_ptr   + prog_id)  # scalar int64

    # ---- gather rows from embedding tables                             
    # word_emb[token_id, h]: stride = H between rows
    word = tl.load(
        word_emb_ptr + token_id * H + h_range,
        mask=mask, other=0.0,
    ).to(tl.float32)

    # pos_emb[pos_id, h]: stride = H between rows
    pos = tl.load(
        pos_emb_ptr + pos_id * H + h_range,
        mask=mask, other=0.0,
    ).to(tl.float32)

    x = word + pos   # bfloat16/float16 → float32 for accumulation

    # ---- LayerNorm (mean & variance over valid elements only) ----------
    x_safe = tl.where(mask, x, 0.0)
    mean   = tl.sum(x_safe, axis=0) / H

    diff   = x - mean
    diff_m = tl.where(mask, diff, 0.0)    # zero out padding to avoid biasing var
    var    = tl.sum(diff_m * diff_m, axis=0) / H
    inv_std = tl.rsqrt(var + 1e-5)
    x_norm  = diff * inv_std

    # ---- apply affine transform ----------------------------------------
    w = tl.load(ln_w_ptr + h_range, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(ln_b_ptr + h_range, mask=mask, other=0.0).to(tl.float32)
    out = x_norm * w + b

    # ---- store (back to bfloat16 if the input was bfloat16) -----------
    tl.store(out_ptr + prog_id * H + h_range, out, mask=mask)


@torch.fx.wrap
def fused_emb_ln_768(x0, x4, x5, x3, x2, x1):
    """
    x0: input_ids   [B, S]
    x4: word_emb    [vocab, 768]
    x5: position_ids [B, S]
    x3: pos_emb     [vocab,  768]
    x2: ln_weight   [768]
    x1: ln_bias     [768]
    Returns:  [B, S, 768]
    """
    B  = x0.shape[0]
    S  = x0.shape[1]
    H  = 768
    HW = B * S

    # Allocate output directly in 3-D shape — no reshape/view needed.
    out = torch.empty((B, S, H), dtype=x4.dtype, device=x4.device)

    # x0 and x5 are contiguous [B, S] int64: flat offset for token b*S+s is prog_id.
    # word_emb / pos_emb / ln_weight / ln_bias are contiguous along the vocab dim.
    _fused_emb_ln_kernel[(HW,)](
        word_ids_ptr = x0,
        pos_ids_ptr  = x5,
        word_emb_ptr = x4,
        pos_emb_ptr  = x3,
        ln_w_ptr     = x2,
        ln_b_ptr     = x1,
        out_ptr      = out,
        H=H,
        seq_len=S,
    )

    return out


def replacement_func():
    return fused_emb_ln_768