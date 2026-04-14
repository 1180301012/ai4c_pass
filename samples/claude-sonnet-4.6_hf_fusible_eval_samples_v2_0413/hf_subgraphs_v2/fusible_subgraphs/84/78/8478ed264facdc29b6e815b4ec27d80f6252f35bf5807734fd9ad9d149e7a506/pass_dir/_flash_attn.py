"""
Shared Flash Attention Triton kernel for fused scaled dot-product attention.

Fuses: matmul(Q, K_T) -> scale -> softmax -> dropout(p=0) -> matmul(attn, V)
       -> permute(0,2,1,3) -> contiguous

Input shapes:
  Q   : [B, H, N, D]
  K_T : [B, H, D, S]  (key already transposed)
  V   : [B, H, S, D]

Output shape (permuted, contiguous):
  Out : [B, N, H, D]
"""

import torch
import triton
import triton.language as tl


def _next_pow2_ge16(n):
    p = 16
    while p < n:
        p *= 2
    return p


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64},  num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32},  num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64},  num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 64},  num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 16,  "BLOCK_N": 64},  num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 32},  num_warps=4, num_stages=4),
    ],
    key=["N_CTX_Q", "N_CTX_KV", "BLOCK_DHEAD"],
)
@triton.jit
def _fused_attn_kernel(
    Q, K_T, V, sm_scale, Out,
    # Q:   [B, H, N, D]
    stride_qb, stride_qh, stride_qn, stride_qd,
    # K_T: [B, H, D, S]
    stride_kb, stride_kh, stride_kd, stride_ks,
    # V:   [B, H, S, D]
    stride_vb, stride_vh, stride_vs, stride_vd,
    # Out: [B, N, H, D]  (permuted output layout)
    stride_ob, stride_on, stride_oh, stride_od,
    B, H, N_CTX_Q, N_CTX_KV, ACTUAL_D,
    BLOCK_DHEAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Grid: (ceil(N/BLOCK_M), B*H)
    start_m  = tl.program_id(0)
    off_bh   = tl.program_id(1)
    off_b    = off_bh // H
    off_h    = off_bh  % H

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DHEAD)

    m_mask = offs_m < N_CTX_Q
    d_mask = offs_d < ACTUAL_D

    # ── Load Q block [BLOCK_M, BLOCK_DHEAD] ──────────────────────────────────
    Q_base = Q + off_b * stride_qb + off_h * stride_qh
    q = tl.load(
        Q_base + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd,
        mask=m_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    # ── Accumulators ─────────────────────────────────────────────────────────
    m_i  = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i  = tl.zeros([BLOCK_M],              dtype=tl.float32)
    acc  = tl.zeros([BLOCK_M, BLOCK_DHEAD], dtype=tl.float32)

    K_base = K_T + off_b * stride_kb + off_h * stride_kh
    V_base = V   + off_b * stride_vb + off_h * stride_vh

    # ── Loop over KV blocks ───────────────────────────────────────────────────
    for start_n in range(0, N_CTX_KV, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N_CTX_KV

        # Load K_T block [BLOCK_DHEAD, BLOCK_N]  (K stored as [B,H,D,S])
        k = tl.load(
            K_base + offs_d[:, None] * stride_kd + offs_n[None, :] * stride_ks,
            mask=d_mask[:, None] & n_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        # Attention scores [BLOCK_M, BLOCK_N]
        qk = tl.dot(q, k) * sm_scale
        # Mask out-of-bound positions
        qk = tl.where(m_mask[:, None] & n_mask[None, :], qk, float("-inf"))

        # Online softmax update
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_new)
        p     = tl.exp(qk - m_new[:, None])

        # Load V block [BLOCK_N, BLOCK_DHEAD]
        v = tl.load(
            V_base + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd,
            mask=n_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        # Accumulate
        acc  = acc * alpha[:, None] + tl.dot(p, v)
        l_i  = l_i * alpha + tl.sum(p, 1)
        m_i  = m_new

    # ── Normalize ─────────────────────────────────────────────────────────────
    acc = acc / l_i[:, None]

    # ── Store output in permuted [B, N, H, D] layout ─────────────────────────
    O_base = Out + off_b * stride_ob + off_h * stride_oh
    tl.store(
        O_base + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od,
        acc.to(Out.dtype.element_ty),
        mask=m_mask[:, None] & d_mask[None, :],
    )


def _fused_attn_fwd(q, k_t, v, sm_scale):
    """
    q   : [B, H, N, D]
    k_t : [B, H, D, S]   (transposed key)
    v   : [B, H, S, D]
    sm_scale : float = 1.0 / scale_factor

    Returns contiguous [B, N, H, D] tensor (equivalent to
    matmul_result.permute(0,2,1,3).contiguous()).
    """
    B, H, N, D = q.shape
    S = k_t.shape[-1]
    BLOCK_DHEAD = _next_pow2_ge16(D)

    # Allocate output in permuted [B, N, H, D] layout
    out = torch.empty((B, N, H, D), dtype=q.dtype, device=q.device)

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_M"]), B * H)

    _fused_attn_kernel[grid](
        q, k_t, v, sm_scale, out,
        q.stride(0),   q.stride(1),   q.stride(2),   q.stride(3),
        k_t.stride(0), k_t.stride(1), k_t.stride(2), k_t.stride(3),
        v.stride(0),   v.stride(1),   v.stride(2),   v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, H, N, S, D,
        BLOCK_DHEAD=BLOCK_DHEAD,
    )
    return out


# ── Shared dispatch wrapper (sm_scale passed as float constant) ──────────────
@torch.fx.wrap
def flash_attn_dispatch(q, k_t, v, sm_scale):
    """Shared dispatch entry-point used by all pass replacement_func()s.
    sm_scale is a float constant embedded by each pass's replacement_args."""
    return _fused_attn_fwd(q, k_t, v, sm_scale)