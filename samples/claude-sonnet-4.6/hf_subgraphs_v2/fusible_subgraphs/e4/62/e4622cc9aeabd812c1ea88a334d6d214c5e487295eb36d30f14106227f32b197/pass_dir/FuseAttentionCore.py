import torch
import triton
import triton.language as tl
import math


# ============================================================
# Flash Attention kernel: Q[B,M,D] @ K^T[B,N,D] → softmax → @V[B,N,D]
# Handles non-contiguous Q/K/V via explicit strides
# Scale is applied to Q inside the kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
    ],
    key=['B', 'M', 'N'],
)
@triton.jit
def flash_attn_bhsd_kernel(
    Q, K, V, Out,
    stride_qb, stride_qm, stride_qd,
    stride_kb, stride_kn, stride_kd,
    stride_vb, stride_vn, stride_vd,
    stride_ob, stride_om, stride_od,
    B, M, N,
    scale,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Computes: Out[b, m, :] = softmax(scale * Q[b,m,:] @ K[b,:,:].T) @ V[b,:,:]
    Grid: (cdiv(M, BLOCK_M), B)
    """
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    q_mask = offs_m[:, None] < M
    # Load and scale Q block
    q = tl.load(
        Q + pid_b * stride_qb + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        mask=q_mask, other=0.0,
    ).to(tl.float32) * scale

    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M],              dtype=tl.float32)
    acc  = tl.zeros([BLOCK_M, HEAD_DIM],   dtype=tl.float32)

    for n_start in range(0, N, BLOCK_N):
        offs_n  = n_start + tl.arange(0, BLOCK_N)
        kv_mask = offs_n[:, None] < N

        k = tl.load(
            K + pid_b * stride_kb + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
            mask=kv_mask, other=0.0,
        ).to(tl.float32)
        v = tl.load(
            V + pid_b * stride_vb + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
            mask=kv_mask, other=0.0,
        ).to(tl.float32)

        # Attention scores [BLOCK_M, BLOCK_N]
        qk = tl.dot(q, tl.trans(k))
        qk = tl.where(offs_n[None, :] < N, qk, float('-inf'))

        # Online softmax update
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        p     = tl.exp(qk - m_new[:, None])
        l_new = l_i * tl.exp(m_i - m_new) + tl.sum(p, axis=1)
        acc   = acc * tl.exp(m_i - m_new)[:, None] + tl.dot(p, v)
        m_i   = m_new
        l_i   = l_new

    acc = acc / l_i[:, None]

    tl.store(
        Out + pid_b * stride_ob + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
        acc.to(Out.dtype.element_ty),
        mask=q_mask,
    )


# ============================================================
# Pattern:  q → q*scale → bmm(q_s, k.T) → softmax → dropout → bmm(attn,v)
# This is the core attention subgraph inside decomposed MHA
# ============================================================
def pattern(q, k, v):
    q_s  = q * 0.125                                     # scale = 1/sqrt(64)
    k_t  = k.transpose(-2, -1)
    attn = torch.bmm(q_s, k_t)
    attn = torch.nn.functional.softmax(attn, dim=-1)
    attn = torch.nn.functional.dropout(attn, 0.0, False, False)
    out  = torch.bmm(attn, v)
    return out


def replacement_args(q, k, v):
    return (q, k, v)


# ============================================================
# Optimised replacement
# ============================================================
@torch.fx.wrap
def fused_attn_core(q, k, v):
    """
    Fused attention: softmax(scale * Q @ K^T) @ V
    q, k, v: [B, M, D] with potentially non-contiguous strides
    Returns: [B, M, D] contiguous
    """
    B = q.shape[0]
    M = q.shape[1]
    N = k.shape[1]
    D = q.shape[2]
    scale = 1.0 / math.sqrt(D)

    out = torch.empty(B, M, D, dtype=q.dtype, device=q.device)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), B)
    flash_attn_bhsd_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        B, M, N,
        scale,
        HEAD_DIM=D,
    )
    return out


def replacement_func():
    return fused_attn_core