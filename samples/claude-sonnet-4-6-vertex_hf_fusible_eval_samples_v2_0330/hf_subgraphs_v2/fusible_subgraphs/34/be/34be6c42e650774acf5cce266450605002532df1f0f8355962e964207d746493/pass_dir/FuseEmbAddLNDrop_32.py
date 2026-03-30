import torch
import triton
import triton.language as tl


# ─── Pattern ────────────────────────────────────────────────────────────────
# Same structure as the 768-dim case but with normalized_shape=(32,)

def pattern(in_0, in_4, pos_embed, in_2, in_1):
    tmp_10 = torch.nn.functional.embedding(in_0, in_4, 1, None, 2.0, False, False)
    tmp_16 = tmp_10 + pos_embed
    tmp_17 = torch.nn.functional.layer_norm(tmp_16, (32,), in_2, in_1, 1e-05)
    tmp_18 = torch.nn.functional.dropout(tmp_17, 0.1, False, False)
    return tmp_18


def replacement_args(in_0, in_4, pos_embed, in_2, in_1):
    return (in_0, in_4, pos_embed, in_2, in_1)


# ─── Triton kernel ──────────────────────────────────────────────────────────

@triton.jit
def _emb_add_ln_32_kernel(
    ids_ptr,        # int64  [N]         flattened token IDs
    wemb_ptr,       # fp16/bf16 [V, H]   word embedding table
    pemb_ptr,       # fp16/bf16 [N, H]   position embeddings (flattened)
    weight_ptr,     # fp16/bf16 [H]      layer-norm weight
    bias_ptr,       # fp16/bf16 [H]      layer-norm bias
    out_ptr,        # fp16/bf16 [N, H]   output (flattened)
    N,
    H,
    eps,
    IS_BF16: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)

    token_id = tl.load(ids_ptr + row)

    cols = tl.arange(0, BLOCK_H)
    mask = cols < H          # for BLOCK_H==H this is always True

    # Load and upcast to fp32
    w = tl.load(wemb_ptr + token_id * H + cols, mask=mask, other=0.0).to(tl.float32)
    p = tl.load(pemb_ptr + row      * H + cols, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(weight_ptr          + cols, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr            + cols, mask=mask, other=0.0).to(tl.float32)

    x = w + p

    # Layer-norm (BLOCK_H == H == 32, no masking needed for variance)
    mean = tl.sum(x, axis=0) / H
    x_c  = x - mean
    var  = tl.sum(x_c * x_c, axis=0) / H
    rstd = tl.rsqrt(var + eps)
    out  = x_c * rstd * g + b

    # Store back in original dtype (dropout training=False → identity)
    if IS_BF16:
        tl.store(out_ptr + row * H + cols, out.to(tl.bfloat16), mask=mask)
    else:
        tl.store(out_ptr + row * H + cols, out.to(tl.float16),  mask=mask)


# ─── Python wrapper ──────────────────────────────────────────────────────────

@torch.fx.wrap
def fused_emb_add_ln_drop_32(in_0, in_4, pos_embed, in_2, in_1):
    """
    in_0      : [B, S]     int64  token IDs
    in_4      : [V, 32]    fp16/bf16 word-embedding table
    pos_embed : [B, S, 32] fp16/bf16 position embeddings
    in_2      : [32]       fp16/bf16 layer-norm weight
    in_1      : [32]       fp16/bf16 layer-norm bias
    returns   : [B, S, 32] same dtype
    """
    H      = 32
    BLOCK  = 32          # exact power-of-2 == H, no masking needed
    N      = in_0.numel()
    eps    = 1e-5

    ids  = in_0.reshape(-1)
    pos  = pos_embed.reshape(N, H)
    out  = torch.empty_like(pos)

    is_bf16 = (in_4.dtype == torch.bfloat16)

    _emb_add_ln_32_kernel[(N,)](
        ids, in_4, pos, in_2, in_1, out,
        N, H, eps,
        IS_BF16=is_bf16,
        BLOCK_H=BLOCK,
    )

    return out.reshape_as(pos_embed)


# ─── replacement_func ────────────────────────────────────────────────────────

def replacement_func():
    return fused_emb_add_ln_drop_32