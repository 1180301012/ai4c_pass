import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    tmp_0 = torch.matmul(in_0, in_1)
    tmp_1 = tmp_0 * 1.0
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1, dtype=torch.float32)
    tmp_3 = tmp_2.to(torch.float32)
    tmp_4 = torch.nn.functional.dropout(tmp_3, p=0.0, training=False)
    tmp_5 = torch.matmul(tmp_4, in_2)
    tmp_6 = tmp_5.transpose(1, 2)
    tmp_7 = tmp_6.contiguous()
    tmp_8 = tmp_7.reshape(1, 257, 1280)
    tmp_9 = tmp_8.contiguous()
    return (tmp_9,)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_K': 32}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_K': 64}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_K': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_K': 128}, num_warps=4, num_stages=1),
    ],
    key=['seq_len_k'],
)
@triton.jit
def flash_attn_f32_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    seq_len_q, seq_len_k, head_dim, num_heads,
    q_stride_b, q_stride_h, q_stride_s, q_stride_d,
    k_stride_b, k_stride_h, k_stride_r, k_stride_c,
    v_stride_b, v_stride_h, v_stride_s, v_stride_d,
    out_stride_b, out_stride_s,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_head_idx = pid // seq_len_q
    q_idx = pid % seq_len_q
    batch_idx = batch_head_idx // num_heads
    head_idx = batch_head_idx % num_heads

    d_range = tl.arange(0, BLOCK_D)
    d_mask = d_range < head_dim

    q_base = batch_idx * q_stride_b + head_idx * q_stride_h
    k_base = batch_idx * k_stride_b + head_idx * k_stride_h
    v_base = batch_idx * v_stride_b + head_idx * v_stride_h

    # Load Q row [head_dim]
    q_ptrs = q_ptr + q_base + q_idx * q_stride_s + d_range * q_stride_d
    q = tl.load(q_ptrs, mask=d_mask, other=0.0)

    # Initialize accumulators
    m_i = float('-inf')
    l_i = 0.0
    o_i = tl.zeros([BLOCK_D], dtype=tl.float32)

    # Iterate over K/V blocks
    for k_start in range(0, seq_len_k, BLOCK_K):
        k_range = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_range < seq_len_k

        # Load K block [head_dim, BLOCK_K]
        k_ptrs = k_ptr + k_base + d_range[:, None] * k_stride_r + k_range[None, :] * k_stride_c
        k_mask_2d = d_mask[:, None] & k_mask[None, :]
        k_block = tl.load(k_ptrs, mask=k_mask_2d, other=0.0)

        # Compute attention scores Q @ K^T -> [BLOCK_K]
        s_ij = tl.sum(q[:, None] * k_block, axis=0)

        # Online softmax
        s_safe = tl.where(k_mask, s_ij, float('-inf'))
        m_ij = tl.max(s_safe, axis=0)
        m_new = tl.maximum(m_i, m_ij)

        alpha = tl.exp(m_i - m_new)
        p_ij = tl.exp(s_ij - m_new)
        p_ij = tl.where(k_mask, p_ij, 0.0)

        l_ij = tl.sum(p_ij, axis=0)
        l_new = alpha * l_i + l_ij

        # Load V block [BLOCK_K, head_dim]
        v_ptrs = v_ptr + v_base + k_range[:, None] * v_stride_s + d_range[None, :] * v_stride_d
        v_mask_2d = k_mask[:, None] & d_mask[None, :]
        v_block = tl.load(v_ptrs, mask=v_mask_2d, other=0.0)

        # Accumulate: O = alpha * O + P @ V
        o_i = alpha * o_i + tl.sum(p_ij[:, None] * v_block, axis=0)

        m_i = m_new
        l_i = l_new

    # Final normalization
    o_i = o_i / l_i

    # Store output in transposed+reshaped layout: [batch, seq_q, heads*head_dim]
    out_offset = batch_idx * out_stride_b + q_idx * out_stride_s + head_idx * head_dim
    out_ptrs = out_ptr + out_offset + d_range
    tl.store(out_ptrs, o_i, mask=d_mask)


@torch.fx.wrap
def flash_attn_f32(q, k, v):
    batch_size = q.shape[0]
    num_heads = q.shape[1]
    seq_len_q = q.shape[2]
    head_dim = q.shape[3]
    seq_len_k = k.shape[3]

    out = torch.empty((batch_size, seq_len_q, num_heads * head_dim), dtype=q.dtype, device=q.device)

    BLOCK_D = 128  # Power of 2, >= head_dim(80)

    grid = (batch_size * num_heads * seq_len_q,)

    flash_attn_f32_kernel[grid](
        q, k, v, out,
        seq_len_q, seq_len_k, head_dim, num_heads,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1),
        BLOCK_D=BLOCK_D,
    )

    return (out,)


def replacement_func():
    return flash_attn_f32