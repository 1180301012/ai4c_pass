"""
Fused pass: 3x Embedding + Add + Dropout(no-op) + LayerNorm for hidden_dim=768
Returns only the LayerNorm output (bigbird / roberta-style graphs).
"""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_H': 1024}, num_warps=8),
        triton.Config({'BLOCK_H': 2048}, num_warps=16),
    ],
    key=['H'],
)
@triton.jit
def _fused_emb_ln_768_only_kernel(
    idx1_ptr, idx2_ptr, idx3_ptr,
    emb1_ptr, emb2_ptr, emb3_ptr,
    gamma_ptr, beta_ptr,
    ln_out_ptr,
    N, H,
    emb1_stride, emb2_stride, emb3_stride,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    mask = cols < H

    idx1 = tl.load(idx1_ptr + row)
    idx2 = tl.load(idx2_ptr + row)
    idx3 = tl.load(idx3_ptr + row)

    e1 = tl.load(emb1_ptr + idx1 * emb1_stride + cols, mask=mask, other=0.0).to(tl.float32)
    e2 = tl.load(emb2_ptr + idx2 * emb2_stride + cols, mask=mask, other=0.0).to(tl.float32)
    e3 = tl.load(emb3_ptr + idx3 * emb3_stride + cols, mask=mask, other=0.0).to(tl.float32)

    x = e1 + e2 + e3

    x_masked = tl.where(mask, x, 0.0)
    mean = tl.sum(x_masked, axis=0) / H
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / H
    inv_std = tl.rsqrt(var + 1e-12)

    gamma = tl.load(gamma_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    beta  = tl.load(beta_ptr  + cols, mask=mask, other=0.0).to(tl.float32)

    out = diff * inv_std * gamma + beta
    tl.store(ln_out_ptr + row * H + cols, out, mask=mask)


@torch.fx.wrap
def fused_emb_ln_768_only(
    word_idx, word_emb,
    token_type_idx, token_type_emb,
    pos_idx, pos_emb,
    ln_weight, ln_bias,
):
    dtype  = word_emb.dtype
    device = word_emb.device
    H      = word_emb.shape[1]

    batch_shape = word_idx.shape
    N = word_idx.numel()

    word_idx_flat = word_idx.reshape(-1)
    tt_idx_flat   = token_type_idx.reshape(-1)

    if pos_idx.numel() < N:
        pos_idx = pos_idx.expand(batch_shape).contiguous()
    pos_idx_flat = pos_idx.reshape(-1)

    ln_out = torch.empty(N, H, dtype=dtype, device=device)

    _fused_emb_ln_768_only_kernel[(N,)](
        word_idx_flat, tt_idx_flat, pos_idx_flat,
        word_emb, token_type_emb, pos_emb,
        ln_weight, ln_bias,
        ln_out,
        N, H,
        word_emb.stride(0), token_type_emb.stride(0), pos_emb.stride(0),
    )

    out_shape = list(batch_shape) + [H]
    return ln_out.reshape(out_shape)


def pattern(word_idx, word_emb, token_type_idx, token_type_emb,
            pos_idx, pos_emb, ln_weight, ln_bias):
    e1 = torch.nn.functional.embedding(word_idx, word_emb, 0, None, 2.0, False, False)
    e2 = torch.nn.functional.embedding(token_type_idx, token_type_emb, None, None, 2.0, False, False)
    s  = e1 + e2
    e3 = torch.nn.functional.embedding(pos_idx, pos_emb, None, None, 2.0, False, False)
    s += e3
    d  = torch.nn.functional.dropout(s, 0.1, False, False)
    ln = torch.nn.functional.layer_norm(d, (768,), ln_weight, ln_bias, 1e-12)
    return ln


def replacement_args(word_idx, word_emb, token_type_idx, token_type_emb,
                     pos_idx, pos_emb, ln_weight, ln_bias):
    return (word_idx, word_emb, token_type_idx, token_type_emb,
            pos_idx, pos_emb, ln_weight, ln_bias)


def replacement_func():
    return fused_emb_ln_768_only