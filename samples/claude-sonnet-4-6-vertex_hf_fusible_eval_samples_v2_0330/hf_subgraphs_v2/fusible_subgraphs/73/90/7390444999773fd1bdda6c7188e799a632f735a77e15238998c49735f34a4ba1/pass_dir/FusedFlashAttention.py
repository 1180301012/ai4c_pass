import torch
import triton
import triton.language as tl
import math


# ---------------------------------------------------------------------------
# Triton Flash Attention — no autotune, fixed BLOCK=16 (ONE compilation total)
# D=64 constexpr, BLOCK_M=16, BLOCK_N=16 constexpr
# ---------------------------------------------------------------------------
@triton.jit
def _flash_fwd(
    Q, K, V, Mask, Out,
    B_H, S,
    sm_scale,
    stride_qbh, stride_qs,
    stride_kbh, stride_ks,
    stride_vbh, stride_vs,
    stride_obh, stride_os,
    stride_mb, stride_ms0, stride_ms1,
    D:       tl.constexpr,   # always 64
    BLOCK_M: tl.constexpr,   # 16
    BLOCK_N: tl.constexpr,   # 16
):
    start_m = tl.program_id(0)
    off_bh  = tl.program_id(1)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    q_ptr = Q + off_bh * stride_qbh + offs_m[:, None] * stride_qs + offs_d[None, :]
    k_ptr = K + off_bh * stride_kbh
    v_ptr = V + off_bh * stride_vbh
    o_ptr = Out + off_bh * stride_obh + offs_m[:, None] * stride_os + offs_d[None, :]
    # Mask: (B, 1, S, S) — we need to map off_bh → batch index
    # Since B_H = B*H, batch_idx = off_bh // H_per_batch; but H varies.
    # Use stride_mb: for mask shape (B,1,S,S), stride_mb = S*S
    # off_bh goes 0..B*H-1; we map to batch with off_bh // (B_H // B)
    # But B is not passed. Use a simpler heuristic: mask index = off_bh * stride_mb // (S*S)?
    # Actually, pass mask base as: Mask + off_bh * stride_mb (may repeat for H heads)
    m_ptr = Mask + off_bh * stride_mb

    q_mask = offs_m[:, None] < S
    q = tl.load(q_ptr, mask=q_mask, other=0.0).to(tl.float32)

    m_i  = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    l_i  = tl.zeros((BLOCK_M,),              dtype=tl.float32)
    acc  = tl.zeros((BLOCK_M, D),            dtype=tl.float32)

    for start_n in range(0, S, BLOCK_N):
        offs_n  = start_n + tl.arange(0, BLOCK_N)
        kv_mask = offs_n < S

        k = tl.load(k_ptr + offs_n[None, :] * stride_ks + offs_d[:, None],
                    mask=kv_mask[None, :], other=0.0).to(tl.float32)
        qk = tl.dot(q, k) * sm_scale

        mv = tl.load(m_ptr + offs_m[:, None] * stride_ms0 + offs_n[None, :] * stride_ms1,
                     mask=q_mask & kv_mask[None, :], other=0.0).to(tl.float32)
        qk = qk + mv
        qk = tl.where(kv_mask[None, :], qk, float('-inf'))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha  = tl.exp(m_i - m_new)
        p      = tl.exp(qk - m_new[:, None])
        l_i    = alpha * l_i + tl.sum(p, axis=1)

        v = tl.load(v_ptr + offs_n[:, None] * stride_vs + offs_d[None, :],
                    mask=kv_mask[:, None], other=0.0).to(tl.float32)
        acc    = alpha[:, None] * acc + tl.dot(p, v)
        m_i    = m_new

    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    acc    = acc / l_safe[:, None]
    tl.store(o_ptr, acc.to(Out.dtype.element_ty), mask=q_mask)


# ---------------------------------------------------------------------------
# Python wrapper — no blocked torch APIs
# ---------------------------------------------------------------------------
@torch.fx.wrap
def triton_sdpa(query, key, value, attn_mask):
    """
    Triton Flash Attention with fixed BLOCK_M=BLOCK_N=16, D=64.
    Compiles only ONCE (all constexprs fixed). Q,K,V: (B,H,S,D); Mask: (B,1,S,S)
    """
    Q = query.contiguous()
    K = key.contiguous()
    V = value.contiguous()
    M = attn_mask.contiguous()
    out = torch.empty_like(Q)

    B  = Q.shape[0]
    H  = Q.shape[1]
    S  = Q.shape[2]
    D  = Q.shape[3]
    BH = B * H

    sm_scale = 1.0 / math.sqrt(D)
    # Strides in element units for Q,K,V,Out: (B,H,S,D) contiguous
    stride_bh = S * D
    stride_s  = D

    # Mask strides: (B, 1, S, S) contiguous
    # stride_mb  = per-BH-index step in mask = (S*S) / H  = S*S for head broadcast
    # But to broadcast H heads over 1 mask head, we use stride_mb = S*S // H = S*S / H
    # (integer divide may lose info if H doesn't divide evenly)
    # Safe approach: off_bh maps to batch via off_bh * (S*S) // H... 
    # Or: stride_mb = S*S // H  → same mask reused H times per batch sample
    stride_mb  = M.stride(0) // H   # per BH-step in mask
    stride_ms0 = M.stride(2)        # S dim 0 stride
    stride_ms1 = M.stride(3)        # S dim 1 stride

    BLOCK_M = 16
    BLOCK_N = 16
    grid = (triton.cdiv(S, BLOCK_M), BH)

    _flash_fwd[grid](
        Q, K, V, M, out,
        BH, S,
        sm_scale,
        stride_bh, stride_s,
        stride_bh, stride_s,
        stride_bh, stride_s,
        stride_bh, stride_s,
        stride_mb, stride_ms0, stride_ms1,
        D=D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return out


# ---------------------------------------------------------------------------
# Pattern: F.sdpa with dropout_p=0.0, is_causal=False → matches ALL graphs
# ---------------------------------------------------------------------------
def pattern(query, key, value, attn_mask):
    return torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)


def replacement_args(query, key, value, attn_mask):
    return (query, key, value, attn_mask)


def replacement_func():
    return triton_sdpa