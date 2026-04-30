import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
    ],
    key=['S'],
)
@triton.jit
def sdpa_fused_out_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    mask_ptr,
    out_ptr,
    B,
    H,
    S,
    stride_q0,
    stride_q1,
    stride_q2,
    stride_q3,
    stride_k0,
    stride_k1,
    stride_k2,
    stride_k3,
    stride_v0,
    stride_v1,
    stride_v2,
    stride_v3,
    stride_m0,
    stride_m1,
    stride_m2,
    stride_m3,
    stride_o0,
    stride_o1,
    stride_o2,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = q_ptr + b * stride_q0 + h * stride_q1 + offs_m[:, None] * stride_q2 + offs_d[None, :] * stride_q3
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < S), other=0.0).to(tl.float32)

    m_i = tl.full((BLOCK_M,), -float('inf'), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    for start_n in range(0, S, BLOCK_N):
        cols = start_n + offs_n

        k_ptrs = k_ptr + b * stride_k0 + h * stride_k1 + cols[:, None] * stride_k2 + offs_d[None, :] * stride_k3
        k = tl.load(k_ptrs, mask=(cols[:, None] < S), other=0.0).to(tl.float32)

        qk = tl.dot(q, tl.trans(k)) * sm_scale

        m_ptrs = mask_ptr + b * stride_m0 + offs_m[:, None] * stride_m2 + cols[None, :] * stride_m3
        valid_mask = (offs_m[:, None] < S) & (cols[None, :] < S)
        attn_mask = tl.load(m_ptrs, mask=valid_mask, other=-float('inf')).to(tl.float32)
        qk = qk + attn_mask

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        v_ptrs = v_ptr + b * stride_v0 + h * stride_v1 + cols[:, None] * stride_v2 + offs_d[None, :] * stride_v3
        v = tl.load(v_ptrs, mask=(cols[:, None] < S), other=0.0).to(tl.float32)
        acc = acc + tl.dot(p, v)
        m_i = m_ij

    acc = acc / l_i[:, None]

    out_ptrs = out_ptr + b * stride_o0 + offs_m[:, None] * stride_o1 + (h * BLOCK_D + offs_d[None, :]) * stride_o2
    tl.store(out_ptrs, acc, mask=(offs_m[:, None] < S))


@torch.fx.wrap
def sdpa_fused_out(q, k, v, mask, route):
    B = q.shape[0]
    H = q.shape[1]
    S = q.shape[2]
    D = q.shape[3]

    out = torch.empty((B, S, H * D), device=q.device, dtype=q.dtype)

    grid = lambda meta: (triton.cdiv(S, meta['BLOCK_M']), B * H)
    sdpa_fused_out_kernel[grid](
        q,
        k,
        v,
        mask,
        out,
        B,
        H,
        S,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        mask.stride(0),
        mask.stride(1),
        mask.stride(2),
        mask.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        0.125,
        BLOCK_D=64,
    )
    return out


def replacement_func():
    return sdpa_fused_out