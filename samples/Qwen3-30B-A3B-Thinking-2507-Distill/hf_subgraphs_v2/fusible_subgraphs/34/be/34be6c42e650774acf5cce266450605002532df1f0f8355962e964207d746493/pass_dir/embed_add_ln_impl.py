"""
Shared Triton kernels and dispatcher for fused embedding+add+layernorm passes.
Imported by FuseEmbedAddLayerNorm_768.py and FuseEmbedAddLayerNorm_32.py.
"""
import torch
import triton
import triton.language as tl


# ── H = 768 kernel ───────────────────────────────────────────────────────────
@triton.jit
def _embed_add_ln_768_kernel(
    input_ids_ptr,        # [1, 15] int64
    word_emb_ptr,         # [250002, 768] fp16/bf16
    pos_emb_ptr,          # [514,   768] fp16/bf16
    ln_weight_ptr,        # [768]        fp16/bf16
    ln_bias_ptr,          # [768]        fp16/bf16
    output_ptr,           # [15, 768]    fp16/bf16
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    eps: tl.constexpr,
):
    row = tl.program_id(0)
    token_id  = tl.load(input_ids_ptr + row)
    offs      = tl.arange(0, BLOCK_H)
    mask      = offs < H

    word_emb = tl.load(word_emb_ptr + token_id * H + offs,
                       mask=mask, other=0.0).to(tl.float32)
    pos_idx    = row + 2
    pos_emb    = tl.load(pos_emb_ptr + pos_idx * H + offs,
                         mask=mask, other=0.0).to(tl.float32)

    x = word_emb + pos_emb

    x_sum = tl.sum(x, axis=0)
    mean  = x_sum / H
    x_c   = tl.where(mask, x - mean, 0.0)
    var   = tl.sum(x_c * x_c, axis=0) / H
    x_norm = x_c * tl.rsqrt(var + eps)

    ln_w  = tl.load(ln_weight_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    ln_b  = tl.load(ln_bias_ptr   + offs, mask=mask, other=0.0).to(tl.float32)
    out   = x_norm * ln_w + ln_b

    tl.store(output_ptr + row * H + offs, out, mask=mask)


# ── H = 32 kernel ────────────────────────────────────────────────────────────
@triton.jit
def _embed_add_ln_32_kernel(
    input_ids_ptr,        # [1, 15] int64
    word_emb_ptr,         # [250002, 32] fp16/bf16
    pos_emb_ptr,          # [512,   32]  fp16/bf16
    ln_weight_ptr,        # [32]         fp16/bf16
    ln_bias_ptr,          # [32]         fp16/bf16
    output_ptr,           # [15, 32]     fp16/bf16
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    eps: tl.constexpr,
):
    row = tl.program_id(0)
    token_id  = tl.load(input_ids_ptr + row)
    offs      = tl.arange(0, BLOCK_H)
    mask      = offs < H

    word_emb = tl.load(word_emb_ptr + token_id * H + offs,
                       mask=mask, other=0.0).to(tl.float32)
    pos_idx    = row + 2
    pos_emb    = tl.load(pos_emb_ptr + pos_idx * H + offs,
                         mask=mask, other=0.0).to(tl.float32)

    x = word_emb + pos_emb

    mean  = tl.sum(x, axis=0) / H
    x_c   = x - mean
    var   = tl.sum(x_c * x_c, axis=0) / H
    x_norm = x_c * tl.rsqrt(var + eps)

    ln_w  = tl.load(ln_weight_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    ln_b  = tl.load(ln_bias_ptr   + offs, mask=mask, other=0.0).to(tl.float32)
    out   = x_norm * ln_w + ln_b

    tl.store(output_ptr + row * H + offs, out, mask=mask)


# ── Shared dispatcher (both pass files return this same function) ────────────
@torch.fx.wrap
def fused_embed_add_ln(in_0, in_2, in_1, in_3, in_4, pos_indices, route):
    """
    Fused embedding-lookup + add + layer-norm.
    route: "h768" or "h32"
    in_0:        [1, 15]      int64   – input token ids
    in_2:        [H]          float   – layer-norm weight
    in_1:        [H]          float   – layer-norm bias
    in_3:        [pos, H]     float   – position embedding weight
    in_4:        [vocab, H]   float   – word embedding weight
    pos_indices: [1, 15]      int64   – position indices (pre-computed)
    """
    if route == "h768":
        H      = 768
        BLOCK_H = 1024   # next power-of-2 >= 768
        output = torch.empty(
            (15, H), dtype=in_4.dtype, device=in_4.device
        )
        _embed_add_ln_768_kernel[(15,)](
            input_ids_ptr=in_0,
            word_emb_ptr=in_4,
            pos_emb_ptr=in_3,
            ln_weight_ptr=in_2,
            ln_bias_ptr=in_1,
            output_ptr=output,
            H=H,
            BLOCK_H=BLOCK_H,
            eps=1e-5,
            num_warps=4,
        )
    else:  # route == "h32"
        H      = 32
        BLOCK_H = 32
        output = torch.empty(
            (15, H), dtype=in_4.dtype, device=in_4.device
        )
        _embed_add_ln_32_kernel[(15,)](
            input_ids_ptr=in_0,
            word_emb_ptr=in_4,
            pos_emb_ptr=in_3,
            ln_weight_ptr=in_2,
            ln_bias_ptr=in_1,
            output_ptr=output,
            H=H,
            BLOCK_H=BLOCK_H,
            eps=1e-5,
            num_warps=1,
        )
    return output