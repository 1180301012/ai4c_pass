import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_H': 64,  'num_warps': 2}),
        triton.Config({'BLOCK_H': 64,  'num_warps': 4}),
        triton.Config({'BLOCK_H': 128, 'num_warps': 4}),
        triton.Config({'BLOCK_H': 256, 'num_warps': 4}),
        triton.Config({'BLOCK_H': 512, 'num_warps': 8}),
        triton.Config({'BLOCK_H': 1024,'num_warps': 8}),
        triton.Config({'BLOCK_H': 1024,'num_warps': 16}),
        triton.Config({'BLOCK_H': 768, 'num_warps': 8}),
        triton.Config({'BLOCK_H': 768, 'num_warps': 16}),
    ],
    key=['H'],
)
@triton.jit
def fused_emb_ln_kernel(
    word_indices_ptr,   # [B*S] int64 – flattened input_ids
    type_indices_ptr,   # [B*S] int64 – flattened token_type_ids
    pos_indices_ptr,    # [S]   int64 – flattened position_ids  (shape [1,S])
    word_emb_ptr,       # [V,  H] float
    type_emb_ptr,       # [T,  H] float
    pos_emb_ptr,        # [P,  H] float
    ln_w_ptr,           # [H]     float
    ln_b_ptr,           # [H]     float
    sum_out_ptr,        # [B*S, H] float – output: emb sum (dropout identity)
    ln_out_ptr,         # [B*S, H] float – output: layer norm
    S,
    H: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    col = pid % (S)   # position index within a sequence
    row = pid // (S)  # batch index

    # --- embedding indices ---
    word_idx = tl.load(word_indices_ptr + pid)
    type_idx = tl.load(type_indices_ptr + pid)
    # pos_indices_ptr[col] = pos_idx for this row; base offset = col * H
    pos_idx  = tl.load(pos_indices_ptr + col)

    # offsets within the embedding vectors
    offsets = tl.arange(0, BLOCK_H)
    mask    = offsets < H

    # --- load three rows, then accumulate ---
    base_w = word_idx * H
    base_t = type_idx * H
    base_p = pos_idx  * H   # = col * H (same position across batches)

    w = tl.load(word_emb_ptr + base_w + offsets, mask=mask, other=0.0)
    t = tl.load(type_emb_ptr + base_t + offsets, mask=mask, other=0.0)
    p = tl.load(pos_emb_ptr  + base_p + offsets, mask=mask, other=0.0)

    s = w + t + p   # summed embeddings (dropout is identity → no-op)
    s_f32 = s.to(tl.float32)

    # --- layer norm in float32 ---
    # mean
    s_sum = tl.sum(tl.where(mask, s_f32, 0.0), axis=0)
    mean  = s_sum / H

    # variance
    d      = s_f32 - mean
    d_m    = tl.where(mask, d, 0.0)
    var    = tl.sum(d_m * d_m, axis=0) / H
    rstd   = 1.0 / tl.sqrt(var + eps)

    # affine transform
    w_ln = tl.load(ln_w_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    b_ln = tl.load(ln_b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    ln_val = d_m * rstd * w_ln + b_ln

    # --- store (cast back to original dtype) ---
    tl.store(sum_out_ptr + pid * H + offsets, s_f32.to(s.dtype), mask=mask)
    tl.store(ln_out_ptr  + pid * H + offsets, ln_val.to(s.dtype),  mask=mask)


@torch.fx.wrap
def emb_fused_dispatch(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, route):
    """
    Fused embedding-lookup + sum-accumulate + layer-norm kernel.

    in_0  : [B, S]      – word (input)         indices (int64)
    in_1  : [H]         – layer-norm bias      (float)
    in_2  : [H]         – layer-norm weight    (float)
    in_3  : [V,  H]     – word embedding table (float16/float32/bfloat16)
    in_4  : [B, S]      – token-type indices   (int64)
    in_5  : [P,  H]     – position embedding table (float16/float32/bfloat16)
    in_6  : [B, S]      – token-type indices   (int64)  [redundant / dead-input in some graphs]
    in_7  : [1,  S]     – position indices     (int64)
    route : str         – routing tag for multi-pass support
    """
    B = in_0.shape[0]
    S = in_0.shape[1]
    H = in_3.shape[1]

    # Reshape to 1-D for flat kernel access
    word_indices = in_0.contiguous().view(-1)
    type_indices = in_4.contiguous().view(-1)
    # pos_indices shape [1, S] → [S]
    pos_indices  = in_7.contiguous().view(-1)

    out_sum = torch.empty(B, S, H, dtype=in_3.dtype, device=in_3.device)
    out_ln  = torch.empty(B, S, H, dtype=in_3.dtype, device=in_3.device)

    grid = (B * S,)
    fused_emb_ln_kernel[grid](
        word_indices, type_indices, pos_indices,
        in_3, in_2, in_5,
        in_1,   in_2,           # ln weight, ln bias  (in_2 holds both)
        out_sum, out_ln,
        S, H, 1e-12,
    )

    if route == "tiny_64_double" or route == "bert_1024_double" or \
       route == "mega_768_double":
        return (out_sum, out_ln)
    else:
        return (out_ln,)