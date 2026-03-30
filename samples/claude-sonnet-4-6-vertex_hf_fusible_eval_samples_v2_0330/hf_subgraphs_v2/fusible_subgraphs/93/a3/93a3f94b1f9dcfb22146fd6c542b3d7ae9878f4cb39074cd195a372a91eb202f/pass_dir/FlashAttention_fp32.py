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
    tmp_8 = tmp_7.reshape(1, 257, -1)
    tmp_9 = tmp_8.contiguous()
    return tmp_9


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32}, num_warps=2),
    ],
    key=['M', 'N', 'D'],
)
@triton.jit
def flash_attn_fwd_fp32(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kd, stride_kn,
    stride_vb, stride_vh, stride_vn, stride_vd,
    B, H, M, N, D,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    bh_idx = tl.program_id(0)
    m_block_idx = tl.program_id(1)

    b = bh_idx // H
    h = bh_idx % H

    m_start = m_block_idx * BLOCK_M
    m_range = m_start + tl.arange(0, BLOCK_M)
    d_range = tl.arange(0, BLOCK_D)

    # Load Q block [BLOCK_M, BLOCK_D] — float32
    Q_ptrs = (Q_ptr
              + b * stride_qb
              + h * stride_qh
              + m_range[:, None] * stride_qm
              + d_range[None, :] * stride_qd)
    q_mask = (m_range[:, None] < M) & (d_range[None, :] < D)
    Q = tl.load(Q_ptrs, mask=q_mask, other=0.0)  # float32

    # Accumulators in float32
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    n_blocks = tl.cdiv(N, BLOCK_N)
    for j in range(n_blocks):
        n_start = j * BLOCK_N
        n_range = n_start + tl.arange(0, BLOCK_N)

        # Load K block [BLOCK_D, BLOCK_N] (K is stored as [D, N])
        K_ptrs = (K_ptr
                  + b * stride_kb
                  + h * stride_kh
                  + d_range[:, None] * stride_kd
                  + n_range[None, :] * stride_kn)
        k_mask = (d_range[:, None] < D) & (n_range[None, :] < N)
        K = tl.load(K_ptrs, mask=k_mask, other=0.0)  # float32

        # Attention scores [BLOCK_M, BLOCK_N]
        S = tl.dot(Q, K, out_dtype=tl.float32)

        # Mask out-of-bounds key positions
        S = tl.where(n_range[None, :] < N, S, float('-inf'))

        # Online softmax (flash attention trick)
        m_new = tl.maximum(m_i, tl.max(S, axis=1))
        alpha = tl.exp(m_i - m_new)
        P = tl.exp(S - m_new[:, None])  # float32

        # Load V block [BLOCK_N, BLOCK_D]
        V_ptrs = (V_ptr
                  + b * stride_vb
                  + h * stride_vh
                  + n_range[:, None] * stride_vn
                  + d_range[None, :] * stride_vd)
        v_mask = (n_range[:, None] < N) & (d_range[None, :] < D)
        V = tl.load(V_ptrs, mask=v_mask, other=0.0)  # float32

        # Accumulate: P × V -> float32
        acc = acc * alpha[:, None] + tl.dot(P, V, out_dtype=tl.float32)
        l_i = l_i * alpha + tl.sum(P, axis=1)
        m_i = m_new

    # Normalize
    acc = acc / l_i[:, None]

    # Write output directly in transposed/reshaped layout [B, M, H*D]
    # Out[b, m, h*D + d] = acc[m_local, d]
    out_base = Out_ptr + b * (M * H * D)
    out_ptrs = out_base + m_range[:, None] * (H * D) + h * D + d_range[None, :]
    out_mask = (m_range[:, None] < M) & (d_range[None, :] < D)
    tl.store(out_ptrs, acc, mask=out_mask)


@torch.fx.wrap
def flash_attn_wrapper_fp32(Q, K, V):
    # Q: [B, H, M, D], K: [B, H, D, N], V: [B, H, N, D]  — float32
    B, H, M, D = Q.shape
    N = V.shape[2]

    Out = torch.empty((B, M, H * D), dtype=torch.float32, device=Q.device)

    BLOCK_D = 128  # next power-of-2 >= D=80

    grid = lambda meta: (B * H, triton.cdiv(M, meta['BLOCK_M']))

    flash_attn_fwd_fp32[grid](
        Q, K, V, Out,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        B, H, M, N, D,
        BLOCK_D=BLOCK_D,
    )

    return Out


def replacement_func():
    return flash_attn_wrapper_fp32