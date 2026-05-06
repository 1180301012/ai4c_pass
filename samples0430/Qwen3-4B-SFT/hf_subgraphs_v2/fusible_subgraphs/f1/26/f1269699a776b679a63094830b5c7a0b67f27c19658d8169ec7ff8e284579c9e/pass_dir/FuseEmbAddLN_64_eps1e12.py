import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────────
# Pattern: two embeddings + add + layer_norm + dropout (inference, no-op)
#   hidden dim = 64, norm_eps = 1e-12
# ─────────────────────────────────────────────────────────────────────────────
def pattern(in_0, in_4, in_5, in_3, in_2, in_1):
    tmp_5 = torch.nn.functional.embedding(in_0, in_4, 1, None, 2.0, False, False)
    tmp_6 = torch.nn.functional.embedding(in_5, in_3, 1, None, 2.0, False, False)
    tmp_7 = tmp_5 + tmp_6
    tmp_8 = torch.nn.functional.layer_norm(tmp_7, (64,), in_2, in_1, 1e-12)
    tmp_9 = torch.nn.functional.dropout(tmp_8, 0.1, False, False)
    return tmp_9


def replacement_args(in_0, in_4, in_5, in_3, in_2, in_1):
    return (in_0, in_4, in_5, in_3, in_2, in_1)


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel: fused word-embedding + position-embedding gather + LayerNorm
#
# Each program handles one (batch, seq) token position.
# BLOCK_H is a constexpr ≥ H (power of 2); H is a runtime scalar.
# All arithmetic in float32 for numerical parity with torch.nn.LayerNorm.
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_H': 64},  num_warps=2),
        triton.Config({'BLOCK_H': 64},  num_warps=4),
        triton.Config({'BLOCK_H': 128}, num_warps=4),
        triton.Config({'BLOCK_H': 128}, num_warps=8),
        triton.Config({'BLOCK_H': 256}, num_warps=8),
    ],
    key=['H'],
)
@triton.jit
def _fused_emb_ln_kernel_64(
    word_ids_ptr,
    pos_ids_ptr,
    word_emb_ptr,
    pos_emb_ptr,
    ln_w_ptr,
    ln_b_ptr,
    out_ptr,
    H:           tl.constexpr,   # 64
    BLOCK_H:     tl.constexpr,   # hidden-slice tile (must divide H)
):
    prog_id = tl.program_id(0)
    grid_id = tl.program_id(1)

    tok_id  = tl.load(word_ids_ptr + prog_id)
    pos_id  = tl.load(pos_ids_ptr  + prog_id)

    h_start = grid_id * BLOCK_H
    h_off   = tl.arange(0, BLOCK_H)
    h_range = h_start + h_off
    mask    = h_range < H

    word = tl.load(
        word_emb_ptr + tok_id * H + h_range, mask=mask, other=0.0,
    ).to(tl.float32)
    pos  = tl.load(
        pos_emb_ptr  + pos_id * H + h_range, mask=mask, other=0.0,
    ).to(tl.float32)
    x = word + pos

    # mean + var in one pass: padding elements are 0 → contribute nothing
    mean   = tl.sum(x, axis=0) / H
    x_sh   = tl.where(mask, x, 0.0)
    var    = tl.sum(x_sh * x_sh, axis=0) / H - mean * mean
    x_hat  = (x - mean) * tl.rsqrt(var + 1e-12)

    w = tl.load(ln_w_ptr + h_range, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(ln_b_ptr + h_range, mask=mask, other=0.0).to(tl.float32)
    out = x_hat * w + b

    tl.store(out_ptr + prog_id * H + h_range, out, mask=mask)


@torch.fx.wrap
def fused_emb_ln_64(x0, x4, x5, x3, x2, x1):
    """
    x0: input_ids      [B, S]
    x4: word_emb       [vocab, 64]
    x5: position_ids   [B, S]
    x3: pos_emb        [vocab,  64]
    x2: ln_weight      [64]
    x1: ln_bias        [64]
    Returns:  [B, S, 64]
    """
    B  = x0.shape[0]
    S  = x0.shape[1]
    H  = 64
    HW = B * S

    out = torch.empty((B, S, H), dtype=x4.dtype, device=x4.device)

    _fused_emb_ln_kernel_64[(HW, 1)](
        word_ids_ptr = x0,
        pos_ids_ptr  = x5,
        word_emb_ptr = x4,
        pos_emb_ptr  = x3,
        ln_w_ptr     = x2,
        ln_b_ptr     = x1,
        out_ptr      = out,
        H=H,
        BLOCK_H=64,
    )

    return out


def replacement_func():
    return fused_emb_ln_64