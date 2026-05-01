"""
Shared Flash-Attention-style Triton kernel for all fused-attention passes.

Computes:
    out = (softmax(Q @ K_t / scale) @ V).permute(0,2,1,3).contiguous()

where K_t = K already transposed, i.e. shape [B, H, Dq, N].

Inputs:
    q   : [B, H, M, Dq]
    k   : [B, H, Dq, N]   (already transposed)
    v   : [B, H, N, Dv]
Returns:
    out : [B, M, H, Dv]   (== permute(0,2,1,3).contiguous())
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M':  64, 'BLOCK_N':  64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M':  32, 'BLOCK_N':  64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M':  64, 'BLOCK_N':  32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N':  64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M':  64, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M':  16, 'BLOCK_N':  64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M':  16, 'BLOCK_N':  32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M':  16, 'BLOCK_N':  16}, num_warps=2, num_stages=1),
        triton.Config({'BLOCK_M':  32, 'BLOCK_N':  32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M':  64, 'BLOCK_N':  64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N':  64}, num_warps=8, num_stages=3),
    ],
    key=['B', 'H', 'M', 'N', 'BLOCK_DQ', 'BLOCK_DV'],
)
@triton.jit
def _fused_attn_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    B, H, M, N,
    # Q strides  [B, H, M, Dq]
    stride_qb, stride_qh, stride_qm, stride_qd,
    # K strides  [B, H, Dq, N]  (already transposed)
    stride_kb, stride_kh, stride_kk, stride_kn,
    # V strides  [B, H, N, Dv]
    stride_vb, stride_vh, stride_vn, stride_vd,
    # Out strides [B, M, H, Dv]
    stride_ob, stride_om, stride_oh, stride_od,
    Dq, Dv,          # actual head dimensions (runtime)
    scale_inv,       # 1.0 / scale  (runtime float)
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_M:  tl.constexpr,
    BLOCK_N:  tl.constexpr,
    BLOCK_DQ: tl.constexpr,   # next_power_of_2(Dq)
    BLOCK_DV: tl.constexpr,   # next_power_of_2(Dv)
):
    pid_m  = tl.program_id(0)
    pid_bh = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh  % H

    m_start = pid_m * BLOCK_M
    offs_m  = m_start + tl.arange(0, BLOCK_M)
    offs_dq = tl.arange(0, BLOCK_DQ)
    offs_dv = tl.arange(0, BLOCK_DV)

    # ------------------------------------------------------------------
    # Load Q  [BLOCK_M, BLOCK_DQ]  in fp32
    # ------------------------------------------------------------------
    q_base = Q_ptr + b * stride_qb + h * stride_qh
    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_dq[None, :] * stride_qd
    q_mask = (offs_m[:, None] < M) & (offs_dq[None, :] < Dq)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)

    # ------------------------------------------------------------------
    # Online-softmax accumulators
    # ------------------------------------------------------------------
    m_i  = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i  = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc  = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)

    k_base = K_ptr + b * stride_kb + h * stride_kh
    v_base = V_ptr + b * stride_vb + h * stride_vh

    n_blocks = tl.cdiv(N, BLOCK_N)

    # ------------------------------------------------------------------
    # Iterate over K / V blocks
    # ------------------------------------------------------------------
    for blk in range(n_blocks):
        n_start = blk * BLOCK_N
        offs_n  = n_start + tl.arange(0, BLOCK_N)
        n_mask  = offs_n < N

        # Load K [BLOCK_DQ, BLOCK_N]
        k_ptrs = k_base + offs_dq[:, None] * stride_kk + offs_n[None, :] * stride_kn
        k_mask = (offs_dq[:, None] < Dq) & n_mask[None, :]
        k = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        # Attention scores [BLOCK_M, BLOCK_N]
        scores = tl.dot(q, k, allow_tf32=True) * scale_inv

        # Mask out-of-bounds keys and query rows
        scores = tl.where(
            (offs_m[:, None] < M) & n_mask[None, :],
            scores,
            float('-inf'),
        )

        # ---- Online softmax update -----------------------------------
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp(m_i - m_new)
        p     = tl.exp(scores - m_new[:, None])
        p     = tl.where(n_mask[None, :], p, 0.0)
        l_i   = alpha * l_i + tl.sum(p, axis=1)

        # Load V [BLOCK_N, BLOCK_DV]
        v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_dv[None, :] * stride_vd
        v_mask = n_mask[:, None] & (offs_dv[None, :] < Dv)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0).to(tl.float32)

        # Accumulate:  acc = alpha * acc + p @ V
        acc   = alpha[:, None] * acc + tl.dot(p, v, allow_tf32=True)
        m_i   = m_new

    # ------------------------------------------------------------------
    # Normalize
    # ------------------------------------------------------------------
    out_f32 = acc / l_i[:, None]

    # Convert back to original dtype
    if IS_FP16:
        out_val = out_f32.to(tl.float16)
    elif IS_BF16:
        out_val = out_f32.to(tl.bfloat16)
    else:
        out_val = out_f32

    # ------------------------------------------------------------------
    # Store to Out [B, M, H, Dv]
    # ------------------------------------------------------------------
    out_ptrs = (Out_ptr
                + b             * stride_ob
                + offs_m[:, None] * stride_om
                + h             * stride_oh
                + offs_dv[None, :] * stride_od)
    out_mask = (offs_m[:, None] < M) & (offs_dv[None, :] < Dv)
    tl.store(out_ptrs, out_val, mask=out_mask)


# ---------------------------------------------------------------------------
# Python wrapper  (decorated with @torch.fx.wrap so FX treats it as a leaf)
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_attn(q, k, v, scale):
    """
    q : [B, H, M, Dq]
    k : [B, H, Dq, N]   (K already transposed)
    v : [B, H, N,  Dv]
    Returns [B, M, H, Dv]  (same result as permute(0,2,1,3).contiguous())
    """
    B, H, M, Dq = q.shape
    _,  _, _,  N  = k.shape
    _,  _, _,  Dv = v.shape

    BLOCK_DQ = triton.next_power_of_2(Dq)
    BLOCK_DV = triton.next_power_of_2(Dv)

    out = torch.empty((B, M, H, Dv), dtype=q.dtype, device=q.device)

    is_fp16 = (q.dtype == torch.float16)
    is_bf16 = (q.dtype == torch.bfloat16)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), B * H)

    _fused_attn_kernel[grid](
        q, k, v, out,
        B, H, M, N,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        Dq, Dv,
        1.0 / float(scale),
        IS_FP16=is_fp16,
        IS_BF16=is_bf16,
        BLOCK_DQ=BLOCK_DQ,
        BLOCK_DV=BLOCK_DV,
    )
    return out