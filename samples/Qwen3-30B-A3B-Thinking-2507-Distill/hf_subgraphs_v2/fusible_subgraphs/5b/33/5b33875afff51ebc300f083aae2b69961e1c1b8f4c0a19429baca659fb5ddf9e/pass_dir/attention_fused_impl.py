"""
Shared Triton kernel for fused attention output computation.
Fuses: scale * scores + mask -> softmax -> dropout(noop) -> matmul -> permute -> contiguous
"""
import torch
import triton
import triton.language as tl


@triton.jit
def fused_attn_output_kernel(
    scores_ptr,    # [B, H, query_len, key_len]  - attention scores (fp16/bf16)
    mask_ptr,      # [1, 1, 1, key_len]           - attention mask (fp16/bf16)
    values_ptr,    # [B, H, key_len, head_dim]    - value vectors (fp16/bf16)
    output_ptr,    # [B, query_len, H, head_dim]  - output (fp16/bf16)
    # Strides for scores
    s_b, s_h, s_q, s_k,
    # Strides for values
    v_b, v_h, v_k, v_d,
    # Strides for output
    o_b, o_q, o_h, o_d,
    # Problem dimensions
    query_len, key_len, head_dim, H,
    # Fused scale parameter
    scale,
    # Block sizes (constexpr for compiler specialisation)
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Each program handles one (b, h, q) triplet.
    Computes: softmax(scale * scores[b,h,q,:] + mask[0,0,0,:]) @ values[b,h,:,d]
    and stores the result into output[b, q, h, d].
    """
    pid = tl.program_id(0)

    # Decode (b, h, q) from flat program id
    h_q = pid % (H * query_len)
    b   = pid // (H * query_len)
    h   = h_q // query_len
    q   = h_q % query_len

    # ------------------------------------------------------------------ #
    # Load attention scores for position (b, h, q)                         #
    # ------------------------------------------------------------------ #
    scores_base = b * s_b + h * s_h + q * s_q
    k_offs = tl.arange(0, BLOCK_K)
    key_mask = k_offs < key_len

    scores = tl.load(
        scores_ptr + scores_base + k_offs * s_k,
        mask=key_mask,
        other=0.0,
    ).to(tl.float32)

    # ------------------------------------------------------------------ #
    # Load attention mask for positions 0..BLOCK_K-1                       #
    # ------------------------------------------------------------------ #
    bias = tl.load(
        mask_ptr + k_offs,
        mask=key_mask,
        other=0.0,
    ).to(tl.float32)

    # ------------------------------------------------------------------ #
    # Softmax over attention scores (stable numerics)                     #
    # ------------------------------------------------------------------ #
    scores = scores * scale + bias
    scores = tl.where(key_mask, scores, -float('inf'))
    m_q = tl.max(scores, axis=0)
    e_q = tl.exp(scores - m_q)
    e_q = tl.where(key_mask, e_q, 0.0)
    attn = e_q / tl.sum(e_q, axis=0)   # shape [BLOCK_K]

    # ------------------------------------------------------------------ #
    # Accumulate weighted sum of value vectors                             #
    # ------------------------------------------------------------------ #
    vals_base = b * v_b + h * v_h
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    for k_start in range(0, key_len, BLOCK_K):
        k_idx = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_idx < key_len

        vals = tl.load(
            values_ptr + vals_base + k_idx[:, None] * v_k + tl.arange(0, BLOCK_D)[None, :] * v_d,
            mask=k_mask[:, None],
            other=0.0,
        ).to(tl.float32)

        # Multiply: attention weights broadcast over BLOCK_D
        acc += tl.sum(
            tl.where(k_mask[:, None], attn[:, None, None] * vals, 0.0),
            axis=0,
        )   # shape [BLOCK_D]

    # ------------------------------------------------------------------ #
    # Store result into output[b, q, h, :]                                 #
    # ------------------------------------------------------------------ #
    out_base = b * o_b + q * o_q + h * o_h
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < head_dim

    tl.store(
        output_ptr + out_base + d_offs * o_d,
        acc.to(vals.dtype),
        mask=d_mask,
    )


@torch.fx.wrap
def fused_attn_scale8(scores, mask, values):
    """Fused attention: scale=8.0 variant."""
    B, H, query_len, _ = scores.shape
    _, _, key_len, _   = values.shape
    _, _, _, head_dim  = scores.shape

    output = torch.empty(B, query_len, H, head_dim, dtype=scores.dtype, device=scores.device)

    BLOCK_K = triton.next_power_of_2(key_len)
    BLOCK_D = triton.next_power_of_2(head_dim)

    grid = (B * H * query_len,)

    fused_attn_output_kernel[grid](
        scores, mask, values, output,
        scores.stride(0), scores.stride(1), scores.stride(2), scores.stride(3),
        values.stride(0), values.stride(1), values.stride(2), values.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        query_len, key_len, head_dim, H,
        8.0,
        BLOCK_K=BLOCK_K,
        BLOCK_D=BLOCK_D,
    )
    return output


@torch.fx.wrap
def fused_attn_sqrth(scores, mask, values):
    """Fused attention: scale=sqrt(8) variant."""
    B, H, query_len, _ = scores.shape
    _, _, key_len, _   = values.shape
    _, _, _, head_dim  = scores.shape

    output = torch.empty(B, query_len, H, head_dim, dtype=scores.dtype, device=scores.device)

    BLOCK_K = triton.next_power_of_2(key_len)
    BLOCK_D = triton.next_power_of_2(head_dim)

    grid = (B * H * query_len,)

    fused_attn_output_kernel[grid](
        scores, mask, values, output,
        scores.stride(0), scores.stride(1), scores.stride(2), scores.stride(3),
        values.stride(0), values.stride(1), values.stride(2), values.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        query_len, key_len, head_dim, H,
        2.8284271247461903,
        BLOCK_K=BLOCK_K,
        BLOCK_D=BLOCK_D,
    )
    return output