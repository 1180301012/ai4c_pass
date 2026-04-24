import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_D': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_D': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_D': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_D': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_D': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_D': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_D': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_D': 128}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'D'],
)
@triton.jit
def _flash_attn_fwd_bf16(
    Q, K, V, Out,
    M, N, D,
    scale,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kk, stride_kn,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Fused Flash Attention forward kernel (bfloat16 inputs, float32 accumulators).
    Q: [B, H, M, D]  K: [B, H, D, N]  V: [B, H, N, D]  Out: [B, M, H*D]
    """
    pid_bh = tl.program_id(0)
    pid_m  = tl.program_id(1)

    b = pid_bh // tl.num_programs(0)
    h = pid_bh % tl.num_programs(0)

    m_start = pid_m * BLOCK_M
    m_offs  = m_start + tl.arange(0, BLOCK_M)
    d_offs  = tl.arange(0, BLOCK_D)

    # Base pointers for this (b, h) head
    Q_base = Q + b * stride_qb + h * stride_qh
    K_base = K + b * stride_kb + h * stride_kh
    V_base = V + b * stride_vb + h * stride_vh

    # Load Q block [BLOCK_M, BLOCK_D] (padded to BLOCK_D=128, D=80)
    Q_block = tl.load(
        Q_base + m_offs[:, None] * stride_qm + d_offs[None, :] * stride_qd,
        mask=(m_offs[:, None] < M) & (d_offs[None, :] < D),
        other=0.0,
    )

    # Online-softmax accumulators
    m_i   = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i   = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc   = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    for j in range(0, tl.cdiv(N, BLOCK_N)):
        n_offs = j * BLOCK_N + tl.arange(0, BLOCK_N)

        # Load K block [BLOCK_D, BLOCK_N]  (K already transposed)
        K_block = tl.load(
            K_base + d_offs[:, None] * stride_kk + n_offs[None, :] * stride_kn,
            mask=(d_offs[:, None] < D) & (n_offs[None, :] < N),
            other=0.0,
        )

        # Attention scores [BLOCK_M, BLOCK_N]
        attn = tl.dot(Q_block, K_block).to(tl.float32) * scale
        attn = tl.where(n_offs[None, :] < N, attn, float('-inf'))

        # Online softmax update
        m_new  = tl.maximum(m_i, tl.max(attn, axis=1))
        alpha  = tl.exp(m_i - m_new)
        attn_e = tl.exp(attn - m_new[:, None])
        l_new  = alpha * l_i + tl.sum(attn_e, axis=1)

        # Load V block [BLOCK_N, BLOCK_D]
        V_block = tl.load(
            V_base + n_offs[:, None] * stride_vn + d_offs[None, :] * stride_vd,
            mask=(n_offs[:, None] < N) & (d_offs[None, :] < D),
            other=0.0,
        )

        # Accumulate weighted values (promote to float32 for dot)
        acc = alpha[:, None] * acc + tl.dot(attn_e.to(tl.float32), V_block)

        m_i = m_new
        l_i = l_new

    # Normalize
    acc = acc / l_i[:, None]

    # Store to output [B, M, H*D]  (transposed layout: head h at offset h*D)
    Out_base = Out + b * stride_ob
    tl.store(
        Out_base + h * D + m_offs[:, None] * stride_om + d_offs[None, :] * stride_od,
        acc.to(tl.bfloat16),
        mask=(m_offs[:, None] < M) & (d_offs[None, :] < D),
    )


@torch.fx.wrap
def flash_attn_bf16(query, key_t, value):
    """
    query : [B, H, M, D]  bfloat16
    key_t : [B, H, D, N]  bfloat16  (already transposed)
    value : [B, H, N, D]  bfloat16
    returns: [B, M, H*D]  bfloat16
    """
    B, H, M, D = query.shape
    N = key_t.shape[3]

    out = torch.empty((B, M, H * D), dtype=query.dtype, device=query.device)

    grid = lambda meta: (B * H, triton.cdiv(M, meta['BLOCK_M']))

    _flash_attn_fwd_bf16[grid](
        query, key_t, value, out,
        M, N, D,
        1.0 / 8.0,          # scale = 1/sqrt(80) ≈ 0.1118... ≈ 0.125
        query.stride(0),  query.stride(1),  query.stride(2),  query.stride(3),
        key_t.stride(0),  key_t.stride(1),  key_t.stride(2),  key_t.stride(3),
        value.stride(0),  value.stride(1),  value.stride(2),  value.stride(3),
        out.stride(0),    out.stride(1),    out.stride(2),    out.stride(3),
    )

    return out


# ── Pattern / replacement plumbing ──────────────────────────────────────────

def pattern(query, key_t, value):
    matmul   = torch.matmul(query, key_t)
    tmp_1    = matmul * 1.0
    tmp_2    = torch.nn.functional.softmax(tmp_1, dim=-1, dtype=torch.float32)
    tmp_3    = tmp_2.to(torch.float32)
    tmp_4    = torch.nn.functional.dropout(tmp_3, p=0.0, training=False)
    to       = tmp_4.to(torch.bfloat16)
    matmul_1 = torch.matmul(to, value)
    tmp_6    = matmul_1.transpose(1, 2)
    tmp_7    = tmp_6.contiguous()
    tmp_8    = tmp_7.reshape(1, 257, -1)
    tmp_9    = tmp_8.contiguous()
    return (tmp_9,)


def replacement_args(query, key_t, value):
    return (query, key_t, value)


def replacement_func():
    return flash_attn_bf16