"""
Fused pass: 3 embedding lookups + add + dropout(training=False) + LayerNorm
Hidden dim = 64, returns (emb_sum, ln_out) — matches MegatronBert-tiny style models.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _fused_emb_ln_64b_kernel(
    word_idx_ptr, type_idx_ptr, pos_idx_ptr,
    word_emb_ptr, word_emb_stride,
    type_emb_ptr, type_emb_stride,
    pos_emb_ptr,  pos_emb_stride,
    ln_weight_ptr, ln_bias_ptr,
    out_emb_ptr, out_ln_ptr,
    eps,
    BLOCK_SIZE: tl.constexpr,
    HIDDEN: tl.constexpr,
):
    token_id = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < HIDDEN

    word_idx = tl.load(word_idx_ptr + token_id)
    type_idx = tl.load(type_idx_ptr + token_id)
    pos_idx  = tl.load(pos_idx_ptr  + token_id)

    w_emb = tl.load(word_emb_ptr + word_idx * word_emb_stride + cols, mask=mask, other=0.0)
    t_emb = tl.load(type_emb_ptr + type_idx * type_emb_stride + cols, mask=mask, other=0.0)
    p_emb = tl.load(pos_emb_ptr  + pos_idx  * pos_emb_stride  + cols, mask=mask, other=0.0)

    emb = w_emb + t_emb + p_emb

    tl.store(out_emb_ptr + token_id * HIDDEN + cols, emb, mask=mask)

    x = emb.to(tl.float32)
    mean = tl.sum(x, axis=0) / HIDDEN
    diff = tl.where(mask, x - mean, 0.0)
    var  = tl.sum(diff * diff, axis=0) / HIDDEN
    rstd = 1.0 / tl.sqrt(var + eps)
    normed = diff * rstd

    g = tl.load(ln_weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(ln_bias_ptr   + cols, mask=mask, other=0.0).to(tl.float32)
    ln_out = (normed * g + b).to(emb.dtype)
    tl.store(out_ln_ptr + token_id * HIDDEN + cols, ln_out, mask=mask)


def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    tmp_7  = torch.nn.functional.embedding(in_0, in_3, 0,    None, 2.0, False, False)
    tmp_8  = torch.nn.functional.embedding(in_6, in_2, None, None, 2.0, False, False)
    tmp_9  = tmp_7 + tmp_8
    tmp_10 = torch.nn.functional.embedding(in_7, in_1, None, None, 2.0, False, False)
    tmp_9 += tmp_10
    tmp_12 = torch.nn.functional.dropout(tmp_9, 0.1, False, False)
    tmp_13 = torch.nn.functional.layer_norm(tmp_12, (64,), in_5, in_4, 1e-12)
    return (tmp_12, tmp_13)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7)


@torch.fx.wrap
def fused_emb_ln_64_both(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    # in_0: word_idx, in_1: pos_emb_tbl, in_2: type_emb_tbl, in_3: word_emb_tbl
    # in_4: ln_bias,  in_5: ln_weight,   in_6: type_idx,     in_7: pos_idx
    HIDDEN = 64
    BLOCK_SIZE = 64

    batch, seq = in_0.shape
    n_tokens   = batch * seq

    word_flat = in_0.reshape(-1)
    type_flat = in_6.reshape(-1)
    pos_flat  = in_7.expand(batch, seq).reshape(-1).contiguous()

    dtype  = in_3.dtype
    device = in_3.device
    out_emb = torch.empty(n_tokens, HIDDEN, dtype=dtype, device=device)
    out_ln  = torch.empty(n_tokens, HIDDEN, dtype=dtype, device=device)

    _fused_emb_ln_64b_kernel[(n_tokens,)](
        word_flat, type_flat, pos_flat,
        in_3, in_3.stride(0),
        in_2, in_2.stride(0),
        in_1, in_1.stride(0),
        in_5, in_4,
        out_emb, out_ln,
        eps=1e-12,
        BLOCK_SIZE=BLOCK_SIZE,
        HIDDEN=HIDDEN,
    )

    return (out_emb.reshape(batch, seq, HIDDEN),
            out_ln .reshape(batch, seq, HIDDEN))


def replacement_func():
    return fused_emb_ln_64_both