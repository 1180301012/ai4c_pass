"""
Shared Triton kernel for fused embedding lookup + addition + layer norm.
Used by multiple pass files with different hidden dimensions and output counts.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def fused_embed_add_ln_kernel(
    # Index tensors (int64, shape [B, S] or [1, S])
    ids_word_ptr,   stride_iw_b,  stride_iw_s,
    ids_tok_ptr,    stride_it_b,  stride_it_s,
    ids_pos_ptr,    stride_ip_b,  stride_ip_s,
    # Embedding weight tables
    word_w_ptr,
    tok_w_ptr,
    pos_w_ptr,
    # Layer-norm parameters
    ln_w_ptr,
    ln_b_ptr,
    # Outputs
    out_emb_ptr,   # [B*S, D] — only written when STORE_EMB is True
    out_ln_ptr,    # [B*S, D]
    # Problem dimensions
    B, S,
    D:        tl.constexpr,
    BLOCK_D:  tl.constexpr,   # smallest power-of-2 >= D
    eps,
    STORE_EMB: tl.constexpr,  # True → also write embedding sum
    OUT_DTYPE: tl.constexpr,  # tl.float16 / tl.bfloat16 / tl.float32
):
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S

    # Load integer indices
    word_idx = tl.load(ids_word_ptr + b * stride_iw_b + s * stride_iw_s)
    tok_idx  = tl.load(ids_tok_ptr  + b * stride_it_b + s * stride_it_s)
    pos_idx  = tl.load(ids_pos_ptr  + b * stride_ip_b + s * stride_ip_s)

    # Build offset vector over the hidden dimension
    d_off  = tl.arange(0, BLOCK_D)
    d_mask = d_off < D

    # Load embedding rows, upcast to fp32 for numerics
    we = tl.load(word_w_ptr + word_idx * D + d_off, mask=d_mask, other=0.0).to(tl.float32)
    te = tl.load(tok_w_ptr  + tok_idx  * D + d_off, mask=d_mask, other=0.0).to(tl.float32)
    pe = tl.load(pos_w_ptr  + pos_idx  * D + d_off, mask=d_mask, other=0.0).to(tl.float32)

    x = we + te + pe   # fp32 embedding sum

    # Optionally store the embedding sum (two-output case)
    if STORE_EMB:
        tl.store(out_emb_ptr + pid * D + d_off, x.to(OUT_DTYPE), mask=d_mask)

    # --- Layer norm (fp32 accumulation) ---
    mean = tl.sum(x, axis=0) / D
    diff = tl.where(d_mask, x - mean, 0.0)
    var  = tl.sum(diff * diff, axis=0) / D
    inv_std = 1.0 / tl.sqrt(var + eps)
    x_norm  = diff * inv_std

    ln_w = tl.load(ln_w_ptr + d_off, mask=d_mask, other=1.0).to(tl.float32)
    ln_b = tl.load(ln_b_ptr + d_off, mask=d_mask, other=0.0).to(tl.float32)
    out  = x_norm * ln_w + ln_b

    tl.store(out_ln_ptr + pid * D + d_off, out.to(OUT_DTYPE), mask=d_mask)


def _torch_to_tl_dtype(dtype):
    if dtype == torch.float16:
        return tl.float16
    elif dtype == torch.bfloat16:
        return tl.bfloat16
    else:
        return tl.float32


def run_fused_embed_add_ln(
    word_ids, tok_ids, pos_ids,
    word_w, tok_w, pos_w,
    ln_w, ln_b,
    D, BLOCK_D,
    store_emb,
):
    """
    Launch the fused kernel and return results.

    Returns
    -------
    If store_emb:
        (emb_sum, ln_out)   both shaped [B, S, D]
    Else:
        ln_out               shaped [B, S, D]
    """
    dtype = word_w.dtype
    device = word_w.device

    # word_ids / tok_ids are [B, S]; pos_ids may be [1, S] (broadcast)
    B = word_ids.shape[0]
    S = word_ids.shape[1]
    N = B * S

    out_ln = torch.empty(N, D, dtype=dtype, device=device)

    if store_emb:
        out_emb = torch.empty(N, D, dtype=dtype, device=device)
    else:
        out_emb = out_ln   # dummy — never written

    # Stride for position-id batch dimension (0 if broadcast)
    stride_ip_b = 0 if (pos_ids.shape[0] == 1 and B > 1) else S

    tl_dtype = _torch_to_tl_dtype(dtype)

    grid = (N,)
    fused_embed_add_ln_kernel[grid](
        word_ids, S, 1,
        tok_ids,  S, 1,
        pos_ids,  stride_ip_b, 1,
        word_w, tok_w, pos_w,
        ln_w, ln_b,
        out_emb, out_ln,
        B, S,
        D=D, BLOCK_D=BLOCK_D,
        eps=1e-12,
        STORE_EMB=store_emb,
        OUT_DTYPE=tl_dtype,
    )

    if store_emb:
        return out_emb.reshape(B, S, D), out_ln.reshape(B, S, D)
    else:
        return out_ln.reshape(B, S, D)