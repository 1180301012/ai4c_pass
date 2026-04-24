"""
Pass: FuseAttentionScale8
Fuses: scores/8.0 + mask -> softmax -> dropout(noop) -> matmul -> permute -> contiguous
into a single Triton kernel.
Matches graphs with scale=8.0 (YituTech conv-bert-base).
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel: one program per (b, h, q) triplet
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_K': 16, 'BLOCK_D':  64}, num_warps=2),
        triton.Config({'BLOCK_K': 16, 'BLOCK_D': 128}, num_warps=2),
        triton.Config({'BLOCK_K': 32, 'BLOCK_D':  64}, num_warps=2),
        triton.Config({'BLOCK_K': 32, 'BLOCK_D': 128}, num_warps=4),
        triton.Config({'BLOCK_K': 64, 'BLOCK_D':  64}, num_warps=4),
        triton.Config({'BLOCK_K': 64, 'BLOCK_D': 128}, num_warps=4),
    ],
    key=['query_len', 'key_len', 'head_dim'],
)
@triton.jit
def _fused_attn_k8(
    scores_ptr, mask_ptr, values_ptr, output_ptr,
    s_b, s_h, s_q, s_k,
    v_b, v_h, v_k, v_d,
    o_b, o_q, o_h, o_d,
    query_len, key_len, head_dim, H,
    scale,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    h_q   = pid % (H * query_len)
    b     = pid // (H * query_len)
    h     = h_q // query_len
    q     = h_q % query_len

    # Load scores[b, h, q, :] and mask[0, 0, 0, :]
    k_offs = tl.arange(0, BLOCK_K)
    key_mask = k_offs < key_len
    scores = tl.load(scores_ptr + b * s_b + h * s_h + q * s_q + k_offs * s_k,
                     mask=key_mask, other=0.0).to(tl.float32)
    bias   = tl.load(mask_ptr + k_offs, mask=key_mask, other=0.0).to(tl.float32)

    # Scale + add mask, stable softmax
    scores = tl.where(key_mask, scores * scale + bias, -float('inf'))
    m_q = tl.max(scores, axis=0)
    e_q = tl.exp(scores - m_q)
    e_q = tl.where(key_mask, e_q, 0.0)
    attn = e_q / tl.sum(e_q, axis=0)

    # Weighted sum over value vectors
    vals_base = b * v_b + h * v_h
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for k_start in range(0, key_len, BLOCK_K):
        k_idx = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_idx < key_len
        vals = tl.load(
            values_ptr + vals_base + k_idx[:, None] * v_k + tl.arange(0, BLOCK_D)[None, :] * v_d,
            mask=k_mask[:, None], other=0.0,
        ).to(tl.float32)
        acc += tl.sum(tl.where(k_mask[:, None], attn[:, None, None] * vals, 0.0), axis=0)

    # Store to output[b, q, h, :]
    out_base = b * o_b + q * o_q + h * o_h
    d_offs = tl.arange(0, BLOCK_D)
    tl.store(output_ptr + out_base + d_offs * o_d,
             acc.to(vals.dtype), mask=d_offs < head_dim)


@torch.fx.wrap
def fused_attn_scale8(scores, mask, values):
    B, H, query_len, _ = scores.shape
    _, _, key_len, _   = values.shape
    _, _, _, head_dim  = values.shape
    output = torch.empty(B, query_len, H, head_dim, dtype=scores.dtype, device=scores.device)
    grid = lambda meta: (B * H * query_len,)
    _fused_attn_k8[grid](
        scores, mask, values, output,
        scores.stride(0), scores.stride(1), scores.stride(2), scores.stride(3),
        values.stride(0), values.stride(1), values.stride(2), values.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        query_len, key_len, head_dim, H,
        8.0,
    )
    return output


def pattern(scores, mask, values):
    tmp_0 = scores / 8.0
    tmp_1 = tmp_0 + mask
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.1, False, False)
    matmul = torch.matmul(tmp_3, values)
    tmp_5 = matmul.permute(0, 2, 1, 3)
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(scores, mask, values):
    return (scores, mask, values)


def replacement_func():
    return fused_attn_scale8