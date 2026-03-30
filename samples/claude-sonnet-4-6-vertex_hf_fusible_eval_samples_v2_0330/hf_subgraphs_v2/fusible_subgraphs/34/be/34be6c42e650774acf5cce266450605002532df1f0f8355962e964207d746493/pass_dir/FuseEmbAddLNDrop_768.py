import torch
import triton
import triton.language as tl


# ─── Pattern ────────────────────────────────────────────────────────────────
# Matches: embedding(tokens, word_table) + pos_embed  →  layer_norm  →  dropout
# The position embeddings are treated as a single "pos_embed" input to the
# pattern; the cumsum-based position-index computation that precedes the
# position embedding lookup lives outside this subgraph.

def pattern(in_0, in_4, pos_embed, in_2, in_1):
    tmp_10 = torch.nn.functional.embedding(in_0, in_4, 1, None, 2.0, False, False)
    tmp_16 = tmp_10 + pos_embed
    tmp_17 = torch.nn.functional.layer_norm(tmp_16, (768,), in_2, in_1, 1e-05)
    tmp_18 = torch.nn.functional.dropout(tmp_17, 0.1, False, False)
    return tmp_18


def replacement_args(in_0, in_4, pos_embed, in_2, in_1):
    return (in_0, in_4, pos_embed, in_2, in_1)


# ─── Triton kernel ──────────────────────────────────────────────────────────

@triton.jit
def _emb_add_ln_kernel(
    ids_ptr,        # int64  [N]            flattened token IDs
    wemb_ptr,       # fp16/bf16 [vocab, H]  word embedding table
    pemb_ptr,       # fp16/bf16 [N, H]      position embeddings (flattened)
    weight_ptr,     # fp16/bf16 [H]         layer-norm weight
    bias_ptr,       # fp16/bf16 [H]         layer-norm bias
    out_ptr,        # fp16/bf16 [N, H]      output (flattened)
    N,
    H,
    eps,
    IS_BF16: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)

    # Load token id for this row
    token_id = tl.load(ids_ptr + row)

    cols = tl.arange(0, BLOCK_H)
    mask = cols < H

    # Load embeddings (convert to fp32 for accurate arithmetic)
    w = tl.load(wemb_ptr + token_id * H + cols, mask=mask, other=0.0).to(tl.float32)
    p = tl.load(pemb_ptr + row     * H + cols, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(weight_ptr         + cols, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr           + cols, mask=mask, other=0.0).to(tl.float32)

    # Add word + position embeddings
    x = w + p

    # Layer-norm over H valid elements
    # masked positions contribute 0 → they don't distort mean
    mean = tl.sum(x, axis=0) / H
    x_c  = tl.where(mask, x - mean, 0.0)           # zero out padding lanes
    var  = tl.sum(x_c * x_c, axis=0) / H
    rstd = tl.rsqrt(var + eps)
    out  = x_c * rstd * g + b

    # Store in original dtype (dropout p=0.1, training=False is identity)
    if IS_BF16:
        tl.store(out_ptr + row * H + cols, out.to(tl.bfloat16), mask=mask)
    else:
        tl.store(out_ptr + row * H + cols, out.to(tl.float16),  mask=mask)


# ─── Python wrapper (must be @torch.fx.wrap) ────────────────────────────────

@torch.fx.wrap
def fused_emb_add_ln_drop_768(in_0, in_4, pos_embed, in_2, in_1):
    """
    in_0      : [B, S]      int64  token IDs
    in_4      : [V, 768]    fp16/bf16 word-embedding table
    pos_embed : [B, S, 768] fp16/bf16 position embeddings
    in_2      : [768]       fp16/bf16 layer-norm weight
    in_1      : [768]       fp16/bf16 layer-norm bias
    returns   : [B, S, 768] same dtype
    """
    H      = 768
    BLOCK  = 1024          # next power-of-2 ≥ H
    N      = in_0.numel()  # B * S
    eps    = 1e-5

    ids    = in_0.reshape(-1)
    pos    = pos_embed.reshape(N, H)
    out    = torch.empty_like(pos)

    is_bf16 = (in_4.dtype == torch.bfloat16)

    _emb_add_ln_kernel[(N,)](
        ids, in_4, pos, in_2, in_1, out,
        N, H, eps,
        IS_BF16=is_bf16,
        BLOCK_H=BLOCK,
    )

    return out.reshape_as(pos_embed)


# ─── replacement_func ───────────────────────────────────────────────────────

def replacement_func():
    return fused_emb_add_ln_drop_768