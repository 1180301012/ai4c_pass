"""
Fused pass: 3x Embedding lookup + Add + LayerNorm (H=64, returns both sum and ln_out)
Covers: tiny-MegatronBertForSequenceClassification graphs (H=64, returns (tmp_12, tmp_13))
"""
import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    tmp_7 = torch.nn.functional.embedding(in_0, in_3, 0, None, 2.0, False, False)
    tmp_8 = torch.nn.functional.embedding(in_6, in_2, None, None, 2.0, False, False)
    tmp_9 = tmp_7 + tmp_8
    tmp_10 = torch.nn.functional.embedding(in_7, in_1, None, None, 2.0, False, False)
    tmp_9 += tmp_10
    tmp_11 = tmp_9
    tmp_12 = torch.nn.functional.dropout(tmp_11, 0.1, False, False)
    tmp_13 = torch.nn.functional.layer_norm(tmp_12, (64,), in_5, in_4, 1e-12)
    return tmp_12, tmp_13


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64},  num_warps=2),
        triton.Config({'BLOCK_SIZE': 128},  num_warps=2),
        triton.Config({'BLOCK_SIZE': 128},  num_warps=4),
    ],
    key=['H', 'S', 'B'],
)
@triton.jit
def _fused_emb_ln_h64_kernel(
    idx_ptr,
    type_ptr,
    pos_ptr,
    word_emb_ptr,
    type_emb_ptr,
    pos_emb_ptr,
    ln_weight_ptr,
    ln_bias_ptr,
    out_sum_ptr,
    out_ln_ptr,
    H,
    S,
    B,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    b = row // S
    s = row % S

    h_off = tl.arange(0, BLOCK_SIZE)
    mask  = h_off < H

    word_idx = tl.load(idx_ptr + row)
    type_idx = tl.load(type_ptr + row)
    pos_idx  = tl.load(pos_ptr + s)

    is_pad = (word_idx == 0)

    x_w = tl.load(word_emb_ptr + word_idx * H + h_off, mask=mask, other=0.0)
    x_t = tl.load(type_emb_ptr + type_idx * H + h_off, mask=mask, other=0.0)
    x_p = tl.load(pos_emb_ptr  + pos_idx  * H + h_off, mask=mask, other=0.0)

    x_w = tl.where(is_pad, tl.zeros([BLOCK_SIZE], dtype=x_w.dtype), x_w)
    x_t = tl.where(is_pad, tl.zeros([BLOCK_SIZE], dtype=x_t.dtype), x_t)
    x_p = tl.where(is_pad, tl.zeros([BLOCK_SIZE], dtype=x_p.dtype), x_p)

    x = x_w.to(tl.float32) + x_t.to(tl.float32) + x_p.to(tl.float32)

    tl.store(out_sum_ptr + row * H + h_off, x.to(x_w.dtype), mask=mask)

    mean = tl.sum(x, axis=0) / H

    diff = tl.where(mask, x - mean, tl.zeros([BLOCK_SIZE], dtype=tl.float32))
    var  = tl.sum(diff * diff, axis=0) / H
    rstd = 1.0 / tl.sqrt(var + 1e-12)

    w  = tl.load(ln_weight_ptr + h_off, mask=mask, other=1.0)
    b_ = tl.load(ln_bias_ptr   + h_off, mask=mask, other=0.0)

    out = diff * rstd * w.to(tl.float32) + b_
    tl.store(out_ln_ptr + row * H + h_off, out.to(x_w.dtype), mask=mask)


@torch.fx.wrap
def _fused_emb_ln_h64_both(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    """
    in_0  : [B, S]         int64  – word indices
    in_1  : [pos_vocab,64] float – position embedding weight
    in_2  : [type_vocab,64] float – token-type embedding weight
    in_3  : [word_vocab,64] float – word embedding weight
    in_4  : [64]           float – layer-norm bias
    in_5  : [64]           float – layer-norm weight
    in_6  : [B, S]         int64  – token-type indices
    in_7  : [1, S]         int64  – position indices
    Returns: (embedding_sum [B,S,64], layer_norm_out [B,S,64])
    """
    B, S = in_0.shape
    H    = in_3.shape[1]        # 64

    out_sum = torch.empty((B, S, H), dtype=in_3.dtype, device=in_0.device)
    out_ln  = torch.empty((B, S, H), dtype=in_3.dtype, device=in_0.device)

    total_rows = B * S

    word_idx_flat = in_0.reshape(-1)
    type_idx_flat = in_6.reshape(-1)
    pos_idx_flat  = in_7[0] if in_7.shape[0] == 1 else in_7.reshape(-1)

    _fused_emb_ln_h64_kernel[(total_rows,)](
        word_idx_flat, type_idx_flat, pos_idx_flat,
        in_3, in_2, in_1, in_5, in_4,
        out_sum, out_ln,
        H, S, B,
    )

    return out_sum, out_ln


def replacement_func():
    return _fused_emb_ln_h64_both