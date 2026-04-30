"""
Shared Flash-Attention kernels + routing dispatcher for all SDPA pass files.

Five (scale, head_dim) variants are handled through a single @torch.fx.wrap
dispatcher that routes on a string constant appended to replacement_args().

Scale constants:
  5.656854249492381  = 4*sqrt(2),  head_dim = 32   (most mit-b0 graphs)
  8.0               = 2*sqrt(16),  head_dim = 64   (face-parsing graphs)
  6.928203230275509 = sqrt(48),   head_dim = 48   (tiny-MobileViT)
  6.0               = sqrt(36),   head_dim = 36   (mobilevit-small, x-small)
"""

import torch
import triton
import triton.language as tl

# ─────────────────────────────────────────────────────────────────────────────
# Kernel: scale=5.656..., head_dim=32
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64},  num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64},  num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256}, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256}, num_warps=8),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64},  num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64},  num_warps=8),
    ],
    key=['S', 'K'],
)
@triton.jit
def _sdpa_5656_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    stride_qb, stride_qh, stride_qm, stride_qn,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vk, stride_vd,
    stride_ob, stride_om, stride_oh, stride_od,
    BH, S, K, D,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    bh_idx  = tl.program_id(0)
    m_block = tl.program_id(1)
    b_idx   = bh_idx // tl.num_programs(0)  # H is fixed in wrapper
    h_idx   = bh_idx % tl.num_programs(0)

    m_start = m_block * BLOCK_M
    m_offs  = m_start + tl.arange(0, BLOCK_M)
    d_offs  = tl.arange(0, BLOCK_D)
    m_mask  = m_offs < S

    # Load Q [BLOCK_M, BLOCK_D]
    Q_bh = Q_ptr + b_idx * stride_qb + h_idx * stride_qh
    q = tl.load(
        Q_bh + m_offs[:, None] * stride_qm + d_offs[None, :] * stride_qn,
        mask=m_mask[:, None], other=0.0,
    ).to(tl.float32)

    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    K_bh = K_ptr + b_idx * stride_kb + h_idx * stride_kh
    V_bh = V_ptr + b_idx * stride_vb + h_idx * stride_vh

    num_n_blocks = tl.cdiv(K, BLOCK_N)
    for n_block in range(num_n_blocks):
        n_start = n_block * BLOCK_N
        n_offs  = n_start + tl.arange(0, BLOCK_N)
        n_mask  = n_offs < K

        # Load K [BLOCK_D, BLOCK_N]  (K stored as [B*H, D, K])
        k = tl.load(
            K_bh + d_offs[:, None] * stride_kn + n_offs[None, :] * stride_kk,
            mask=n_mask[None, :], other=0.0,
        ).to(tl.float32)

        qk = tl.dot(q, k) * scale
        qk = tl.where(n_mask[None, :], qk, float('-inf'))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p     = tl.exp(qk - m_new[:, None])
        l_new = alpha * l_i + tl.sum(p, axis=1)

        # Load V [BLOCK_N, BLOCK_D]
        v = tl.load(
            V_bh + n_offs[:, None] * stride_vk + d_offs[None, :] * stride_vd,
            mask=n_mask[:, None], other=0.0,
        ).to(tl.float32)

        acc    += alpha[:, None] * acc
        acc    += tl.dot(p, v)
        m_i     = m_new
        l_i     = l_new

    out = acc / l_i[:, None]

    # Store: out_flat[b*H+h, s, d]  via strides for a [B,S,H,D] tensor
    Out_bh = Out_ptr + bh_idx * stride_ob
    tl.store(
        Out_bh + m_offs[:, None] * stride_om + d_offs[None, :] * stride_od,
        out,
        mask=m_mask[:, None],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Kernel: scale=8.0, head_dim=64
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64},  num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64},  num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256}, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256}, num_warps=8),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64},  num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64},  num_warps=8),
    ],
    key=['S', 'K'],
)
@triton.jit
def _sdpa_80_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    stride_qb, stride_qh, stride_qm, stride_qn,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vk, stride_vd,
    stride_ob, stride_om, stride_oh, stride_od,
    BH, S, K, D,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    bh_idx  = tl.program_id(0)
    m_block = tl.program_id(1)
    b_idx   = bh_idx // (tl.num_programs(0))
    h_idx   = bh_idx % (tl.num_programs(0))

    m_start = m_block * BLOCK_M
    m_offs  = m_start + tl.arange(0, BLOCK_M)
    d_offs  = tl.arange(0, BLOCK_D)
    m_mask  = m_offs < S

    Q_bh = Q_ptr + b_idx * stride_qb + h_idx * stride_qh
    q = tl.load(
        Q_bh + m_offs[:, None] * stride_qm + d_offs[None, :] * stride_qn,
        mask=m_mask[:, None], other=0.0,
    ).to(tl.float32)

    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    K_bh = K_ptr + b_idx * stride_kb + h_idx * stride_kh
    V_bh = V_ptr + b_idx * stride_vb + h_idx * stride_vh

    num_n_blocks = tl.cdiv(K, BLOCK_N)
    for n_block in range(num_n_blocks):
        n_start = n_block * BLOCK_N
        n_offs  = n_start + tl.arange(0, BLOCK_N)
        n_mask  = n_offs < K

        k = tl.load(
            K_bh + d_offs[:, None] * stride_kn + n_offs[None, :] * stride_kk,
            mask=n_mask[None, :], other=0.0,
        ).to(tl.float32)

        qk = tl.dot(q, k) * scale
        qk = tl.where(n_mask[None, :], qk, float('-inf'))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p     = tl.exp(qk - m_new[:, None])
        l_new = alpha * l_i + tl.sum(p, axis=1)

        v = tl.load(
            V_bh + n_offs[:, None] * stride_vk + d_offs[None, :] * stride_vd,
            mask=n_mask[:, None], other=0.0,
        ).to(tl.float32)

        acc    += alpha[:, None] * acc
        acc    += tl.dot(p, v)
        m_i     = m_new
        l_i     = l_new

    out = acc / l_i[:, None]

    Out_bh = Out_ptr + bh_idx * stride_ob
    tl.store(
        Out_bh + m_offs[:, None] * stride_om + d_offs[None, :] * stride_od,
        out,
        mask=m_mask[:, None],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers: allocate output, launch kernel
# ─────────────────────────────────────────────────────────────────────────────

def _launch_5656(query, key_t, value):
    B, H, S, K = query.shape
    D = 32
    query_flat = query.reshape(B * H, S, K)
    out_flat   = torch.empty((B * H, S, D), dtype=query.dtype, device=query.device)
    grid = lambda meta: (B * H, triton.cdiv(S, meta['BLOCK_M']))
    _sdpa_5656_kernel[grid](
        query_flat, key_t, value, out_flat,
        query_flat.stride(0), query_flat.stride(1), query_flat.stride(2), 1,
        key_t.stride(0),        key_t.stride(1),        key_t.stride(2),  1,
        value.stride(0),        value.stride(1),        value.stride(2),  1,
        out_flat.stride(0), out_flat.stride(1), out_flat.stride(2),
        B * H, S, key_t.shape[-1], D,
        5.656854249492381,
        BLOCK_D=D,
    )
    return out_flat.reshape(B, S, H, D)


def _launch_80(query, key_t, value):
    B, H, S, K = query.shape
    D = 64
    query_flat = query.reshape(B * H, S, K)
    out_flat   = torch.empty((B * H, S, D), dtype=query.dtype, device=query.device)
    grid = lambda meta: (B * H, triton.cdiv(S, meta['BLOCK_M']))
    _sdpa_80_kernel[grid](
        query_flat, key_t, value, out_flat,
        query_flat.stride(0), query_flat.stride(1), query_flat.stride(2), 1,
        key_t.stride(0),        key_t.stride(1),        key_t.stride(2),  1,
        value.stride(0),        value.stride(1),        value.stride(2),  1,
        out_flat.stride(0), out_flat.stride(1), out_flat.stride(2),
        B * H, S, key_t.shape[-1], D,
        8.0,
        BLOCK_D=D,
    )
    return out_flat.reshape(B, S, H, D)


def _launch_6928(query, key_t, value):
    """head_dim=48 → BLOCK_D=64 with masking."""
    B, H, S, K = query.shape
    D = 48
    query_flat = query.reshape(B * H, S, K)
    out_flat   = torch.empty((B * H, S, D), dtype=query.dtype, device=query.device)
    grid = lambda meta: (B * H, triton.cdiv(S, meta['BLOCK_M']))
    _sdpa_80_kernel[grid](
        query_flat, key_t, value, out_flat,
        query_flat.stride(0), query_flat.stride(1), query_flat.stride(2), 1,
        key_t.stride(0),        key_t.stride(1),        key_t.stride(2),  1,
        value.stride(0),        value.stride(1),        value.stride(2),  1,
        out_flat.stride(0), out_flat.stride(1), out_flat.stride(2),
        B * H, S, key_t.shape[-1], D,
        6.928203230275509,
        BLOCK_D=64,
    )
    return out_flat.reshape(B, S, H, D)


def _launch_60(query, key_t, value):
    """head_dim=36 → BLOCK_D=64 with masking."""
    B, H, S, K = query.shape
    D = 36
    query_flat = query.reshape(B * H, S, K)
    out_flat   = torch.empty((B * H, S, D), dtype=query.dtype, device=query.device)
    grid = lambda meta: (B * H, triton.cdiv(S, meta['BLOCK_M']))
    _sdpa_80_kernel[grid](
        query_flat, key_t, value, out_flat,
        query_flat.stride(0), query_flat.stride(1), query_flat.stride(2), 1,
        key_t.stride(0),        key_t.stride(1),        key_t.stride(2),  1,
        value.stride(0),        value.stride(1),        value.stride(2),  1,
        out_flat.stride(0), out_flat.stride(1), out_flat.stride(2),
        B * H, S, key_t.shape[-1], D,
        6.0,
        BLOCK_D=64,
    )
    return out_flat.reshape(B, S, H, D)


# ─────────────────────────────────────────────────────────────────────────────
# Single routing dispatcher – returned by replacement_func() in every pass file
# ─────────────────────────────────────────────────────────────────────────────

@torch.fx.wrap
def dispatch_fused_sdpa(query, key_t, value, route):
    if route == "5656":
        return _launch_5656(query, key_t, value)
    elif route == "80":
        return _launch_80(query, key_t, value)
    elif route == "6928":
        return _launch_6928(query, key_t, value)
    else:  # "60"
        return _launch_60(query, key_t, value)