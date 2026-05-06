import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_K': 64},  num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 64},  num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_K': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 256}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_K': 64},  num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_K': 128}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_K': 32},  num_stages=4, num_warps=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 32},  num_stages=4, num_warps=4),
    ],
    key=['S_q', 'S_k', 'HEAD_D'],
)
@triton.jit
def _flash_attn_fused_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    B, H,
    S_q_st, S_k_st,
    K_stride0, K_stride1, K_stride2,
    V_stride0, V_stride1, V_stride2,
    O_stride0, O_stride1, O_stride2,
    scale,
    HEAD_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Flash-attention-style Scaled Dot-Product Attention in a single kernel pass.
    Inputs:
      Q  : [B, S_q, H, HEAD_D]
      K  : [B, H, HEAD_D, S_k]  (already transposed key)
      V  : [B, H, S_k, HEAD_D]
    Output:
      O  : [B, S_q, H*HEAD_D]
    """
    batch_id = tl.program_id(2)
    h_id     = tl.program_id(1)
    m_blk    = tl.program_id(0)

    m_start = m_blk * BLOCK_M
    offs_m  = m_start + tl.arange(0, BLOCK_M)
    offs_d  = tl.arange(0, HEAD_D)

    # Per-head Q base pointers
    Q_b = Q_ptr + batch_id * H * S_q_st * HEAD_D + h_id * S_q_st * HEAD_D
    # Per-head K base (key is already transposed: [HEAD_D × S_k])
    K_b = K_ptr + batch_id * H * HEAD_D * S_k_st + h_id * HEAD_D * S_k_st
    # Per-head V base
    V_b = V_ptr + batch_id * H * S_k_st * HEAD_D + h_id * S_k_st * HEAD_D

    acc = tl.zeros([BLOCK_M, HEAD_D], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Loop over S_k in tiles of BLOCK_K
    for k_start in range(0, S_k_st, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # ── Load query tile [BLOCK_M, HEAD_D] ──────────────────────────────
        q_ptrs = Q_b + offs_m[:, None] * HEAD_D + offs_d[None, :]
        mask_q = (offs_m[:, None] < S_q_st) & (offs_d[None, :] < HEAD_D)
        q = tl.load(q_ptrs, mask=mask_q, other=0.0)

        # ── Load key tile [HEAD_D, BLOCK_K] ────────────────────────────────
        k_ptrs = K_b + offs_d[:, None] * S_k_st + offs_k[None, :]
        mask_k = (offs_k[None, :] < S_k_st) & (offs_d[:, None] < HEAD_D)
        k = tl.load(k_ptrs, mask=mask_k, other=0.0)

        # ── Attention scores [BLOCK_M, BLOCK_K]: s = Q·K^T * scale ─────────
        # Use float32 accumulation for stability (even for fp16 inputs)
        qk  = tl.dot(q.to(tl.float32), k.to(tl.float32)).to(tl.float32) * scale
        # Mask padding in keys
        qk  = tl.where(offs_k[None, :] < S_k_st, qk, float('-inf'))

        # ── Online softmax update ───────────────────────────────────────────
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        p     = tl.exp(qk - m_new[:, None])
        alpha = tl.exp(l_i - m_new)
        l_i   = l_i * alpha + tl.sum(p, axis=1)

        # ── Load value tile [BLOCK_K, HEAD_D] ──────────────────────────────
        v_ptrs = V_b + offs_k[:, None] * HEAD_D + offs_d[None, :]
        mask_v = (offs_k[:, None] < S_k_st) & (offs_d[None, :] < HEAD_D)
        v = tl.load(v_ptrs, mask=mask_v, other=0.0)

        # ── Accumulate: acc = p @ V,  m_i ← m_new,  l_i ← l_i*scale ──────
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float32), v.to(tl.float32)).to(tl.float32)
        m_i = m_new

    # ── Normalize ───────────────────────────────────────────────────────────
    acc = acc / l_i[:, None]

    # ── Write output [B, S_q, H*HEAD_D] at column h*HEAD_D + offs_d ───────
    O_b        = O_ptr + batch_id * S_q_st * H * HEAD_D
    out_offs   = offs_m * H * HEAD_D + h_id * HEAD_D + offs_d
    out_ptrs   = O_b + out_offs
    out_mask   = offs_m < S_q_st
    O_dtype    = O_ptr.dtype.element_ty
    tl.store(out_ptrs, acc.to(O_dtype), mask=out_mask)


@torch.fx.wrap
def flash_attn_wrapper(in_0, in_1, in_2, scale_val, scale_frac):
    """
    Fully-fused Scaled Dot-Product attention replacement.

    in_0 : [B, H, S_q,   D]  – query
    in_1 : [B, H, D,     S_k] – transposed key
    in_2 : [B, H, S_k,  D]   – value
    Returns [B, S_q, H*D] which is exactly what the original view() produced.
    """
    B  = in_0.shape[0]
    H  = in_0.shape[1]
    S_q = in_0.shape[2]
    HEAD_D = in_0.shape[3]

    scale = float(scale_val)

    # Allocate output with the same shape as the original view()
    out = torch.empty(
        (B, S_q, H * HEAD_D), dtype=in_0.dtype, device=in_0.device
    )

    # Strides
    s0_q = in_0.stride(0)
    s1_q = in_0.stride(1)
    s2_q = in_0.stride(2)
    s3_q = in_0.stride(3)

    s0_k = in_1.stride(0)
    s1_k = in_1.stride(1)
    s2_k = in_1.stride(2)
    s3_k = in_1.stride(3)

    s0_v = in_2.stride(0)
    s1_v = in_2.stride(1)
    s2_v = in_2.stride(2)
    s3_v = in_2.stride(3)

    s0_o = out.stride(0)
    s1_o = out.stride(1)
    s2_o = out.stride(2)

    grid = lambda META: (
        triton.cdiv(S_q, META['BLOCK_M']),
        H,
        B,
    )

    _flash_attn_fused_kernel[grid](
        in_0, in_1, in_2, out,
        B, H,
        S_q, in_1.shape[3],        # S_k
        s2_k, s3_k, s0_k, s1_k,   # K strides: (seq, head_dim, batch, head)
        s2_v, s3_v, s0_v, s1_v,   # V strides
        s1_o, s2_o,                # O stride over heads+head_dim
        scale,
        HEAD_D=HEAD_D,
        scale_frac=scale_frac,
    )
    return out