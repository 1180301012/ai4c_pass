"""
Shared Triton kernels and dispatch wrapper used by all passes.
All pass files import `dispatch_fn` from here so they share a single
replacement_func(), satisfying the framework's replacement_func_limit.
"""
import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────────
# Kernel: fused Embedding + LayerNorm  (H = 768, eps = 1e-5)
# Fixed config: BLOCK_SIZE=1024, no autotune → eliminates autotune overhead
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def _emb_ln_768_kernel(
    input_ids_ptr, word_emb_ptr,
    pos_ids_ptr,   pos_emb_ptr,
    ln_weight_ptr, ln_bias_ptr,
    output_ptr,
    n_tokens, eps,
    H:          tl.constexpr,   # 768
    BLOCK_SIZE: tl.constexpr,   # 1024
):
    pid     = tl.program_id(0)
    tok_idx = pid
    if tok_idx >= n_tokens:
        return

    word_idx = tl.load(input_ids_ptr + tok_idx)
    pos_idx  = tl.load(pos_ids_ptr   + tok_idx)

    offsets = tl.arange(0, BLOCK_SIZE)
    mask    = offsets < H

    # word embedding (padding_idx = 1)
    word_emb = tl.load(word_emb_ptr + word_idx * H + offsets, mask=mask, other=0.0)
    is_pad   = (word_idx == 1)
    word_emb = tl.where(is_pad, word_emb * 0.0, word_emb)

    # position embedding
    pos_emb = tl.load(pos_emb_ptr + pos_idx * H + offsets, mask=mask, other=0.0)

    x       = word_emb + pos_emb
    x_f32   = x.to(tl.float32)
    mean    = tl.sum(x_f32, axis=0) / H
    x_cent  = tl.where(mask, x_f32 - mean, 0.0)
    var     = tl.sum(x_cent * x_cent, axis=0) / H
    inv_std = tl.rsqrt(var + eps)
    x_norm  = x_cent * inv_std

    ln_w = tl.load(ln_weight_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    ln_b = tl.load(ln_bias_ptr   + offsets, mask=mask, other=0.0).to(tl.float32)
    out  = (x_norm * ln_w + ln_b).to(x.dtype)
    tl.store(output_ptr + tok_idx * H + offsets, out, mask=mask)


def _run_emb_ln_768(in_0, in_1, in_2, in_3, in_4, in_5):
    B, S = in_0.shape
    H = 768
    n_tok = B * S
    out = torch.empty((B, S, H), dtype=in_4.dtype, device=in_0.device)
    _emb_ln_768_kernel[(n_tok,)](
        in_0, in_4, in_5, in_3, in_2, in_1, out,
        n_tok, 1e-5, H, 1024, num_warps=8,
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Kernel: fused Embedding + LayerNorm  (H = 64, eps = 1e-12)
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def _emb_ln_64_kernel(
    input_ids_ptr, word_emb_ptr,
    pos_ids_ptr,   pos_emb_ptr,
    ln_weight_ptr, ln_bias_ptr,
    output_ptr,
    n_tokens, eps,
    H:          tl.constexpr,   # 64
    BLOCK_SIZE: tl.constexpr,   # 64
):
    pid     = tl.program_id(0)
    tok_idx = pid
    if tok_idx >= n_tokens:
        return

    word_idx = tl.load(input_ids_ptr + tok_idx)
    pos_idx  = tl.load(pos_ids_ptr   + tok_idx)

    offsets = tl.arange(0, BLOCK_SIZE)
    mask    = offsets < H

    word_emb = tl.load(word_emb_ptr + word_idx * H + offsets, mask=mask, other=0.0)
    is_pad   = (word_idx == 1)
    word_emb = tl.where(is_pad, word_emb * 0.0, word_emb)

    pos_emb = tl.load(pos_emb_ptr + pos_idx * H + offsets, mask=mask, other=0.0)

    x       = word_emb + pos_emb
    x_f32   = x.to(tl.float32)
    mean    = tl.sum(x_f32, axis=0) / H
    x_cent  = tl.where(mask, x_f32 - mean, 0.0)
    var     = tl.sum(x_cent * x_cent, axis=0) / H
    inv_std = tl.rsqrt(var + eps)
    x_norm  = x_cent * inv_std

    ln_w = tl.load(ln_weight_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    ln_b = tl.load(ln_bias_ptr   + offsets, mask=mask, other=0.0).to(tl.float32)
    out  = (x_norm * ln_w + ln_b).to(x.dtype)
    tl.store(output_ptr + tok_idx * H + offsets, out, mask=mask)


def _run_emb_ln_64(in_0, in_1, in_2, in_3, in_4, in_5):
    B, S = in_0.shape
    H = 64
    n_tok = B * S
    out = torch.empty((B, S, H), dtype=in_4.dtype, device=in_0.device)
    _emb_ln_64_kernel[(n_tok,)](
        in_0, in_4, in_5, in_3, in_2, in_1, out,
        n_tok, 1e-12, H, 64, num_warps=4,
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Kernel: relative position bias  (B=1 assumed, out has shape [1, S, S])
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def _rel_pos_bias_kernel(
    pos_ids_ptr,   # [1, S] int64  (flat: S elements)
    out_ptr,       # [1, S, S] int64 (flat: S*S elements)
    S,
):
    pid     = tl.program_id(0)          # 0 .. S*S-1
    row_idx = pid // S
    col_idx = pid % S

    pos_i = tl.load(pos_ids_ptr + row_idx).to(tl.int32)
    pos_j = tl.load(pos_ids_ptr + col_idx).to(tl.int32)

    diff    = pos_j - pos_i
    offset  = tl.where(diff < 0, 16, 0).to(tl.int64)
    abs_val = tl.abs(diff)

    floor_val = tl.floor(tl.log(abs_val.to(tl.float32) / 8.0) * (8.0 / 2.772588722239781)).to(tl.int64)
    large_b   = tl.minimum(8 + floor_val, 15)
    bias_val  = tl.where(abs_val < 8, abs_val.to(tl.int64), large_b)
    result    = offset + bias_val

    tl.store(out_ptr + row_idx * S + col_idx, result)


def _run_rel_pos_bias(in_5):
    B, S  = in_5.shape
    out   = torch.empty((B, S, S), dtype=torch.int64, device=in_5.device)
    _rel_pos_bias_kernel[(S * S,)](in_5, out, S)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Single shared dispatch wrapper  (all passes return this same object)
# Fixed 7-arg signature to minimize Python overhead vs *args
# ─────────────────────────────────────────────────────────────────────────────
@torch.fx.wrap
def dispatch_fn(arg0, arg1, arg2, arg3, arg4, arg5, route):
    """
    Route via the last string argument:
      "emb_768"  → fused embedding + LayerNorm (H=768, eps=1e-5)
      "emb_64"   → fused embedding + LayerNorm (H=64,  eps=1e-12)
      "pos_11"   → relative position bias for S=11
      "pos_45"   → relative position bias for S=45
      "pos_7"    → relative position bias for S=7
    """
    if route == "emb_768":
        return _run_emb_ln_768(arg0, arg1, arg2, arg3, arg4, arg5)
    elif route == "emb_64":
        return _run_emb_ln_64(arg0, arg1, arg2, arg3, arg4, arg5)
    elif route == "pos_11":
        return _run_rel_pos_bias(arg0)
    elif route == "pos_45":
        return _run_rel_pos_bias(arg0)
    elif route == "pos_7":
        return _run_rel_pos_bias(arg0)
    # fallback
    return _run_emb_ln_768(arg0, arg1, arg2, arg3, arg4, arg5)