"""
Shared Triton Flash Attention kernel used by all SDPA+Transpose passes.
Computes: output[b, s, h, d] = softmax(Q[b,h,s,:] @ K[b,h,:,s'].T * scale + mask[b,0,s,s']) @ V[b,h,s',:]
Output layout is [B, S, H, D] (already transposed from the standard [B, H, S, D] SDPA output).
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Best throughput for large batches/sequences (primary configs)
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64},  num_stages=5, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64},  num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 16},  num_stages=2, num_warps=4),
    ],
    key=['B', 'H', 'S', 'D'],
)
@triton.jit
def flash_attn_fwd_kernel(
    # Input pointers
    Q_ptr, K_ptr, V_ptr, Mask_ptr,
    # Output pointer (layout [B, S, H, D])
    O_ptr,
    # Strides for Q [B, H, S, D]
    stride_qb, stride_qh, stride_qs, stride_qd,
    # Strides for K [B, H, S, D]
    stride_kb, stride_kh, stride_ks, stride_kd,
    # Strides for V [B, H, S, D]
    stride_vb, stride_vh, stride_vs, stride_vd,
    # Strides for Mask [B, 1, S, S]
    stride_mb, stride_mh, stride_ms, stride_mn,
    # Strides for O [B, S, H, D]
    stride_ob, stride_os, stride_oh, stride_od,
    # Dimensions
    B, H, S, D,
    # Attention scale
    scale,
    # Block sizes (compile-time constants)
    BLOCK_M:   tl.constexpr,
    BLOCK_N:   tl.constexpr,
    HEAD_DIM:  tl.constexpr,
    # Whether to use fp16 (vs fp32) accumulation for the QK product
    USE_FP16:  tl.constexpr,
):
    # Each program handles one (batch, head) pair and one block of M query rows
    off_bh = tl.program_id(0)
    off_b  = off_bh // H
    off_h  = off_bh %  H
    off_m  = tl.program_id(1)

    # Base pointers for this (batch, head)
    Q_base = Q_ptr + off_b * stride_qb + off_h * stride_qh
    K_base = K_ptr + off_b * stride_kb + off_h * stride_kh
    V_base = V_ptr + off_b * stride_vb + off_h * stride_vh
    # Output base: [B, S, H, D] → [b, :, h, :]
    O_base = O_ptr + off_b * stride_ob + off_h * stride_oh

    # Index ranges
    m_range = tl.arange(0, BLOCK_M)
    n_range = tl.arange(0, BLOCK_N)
    d_range = tl.arange(0, HEAD_DIM)

    q_row  = off_m * BLOCK_M + m_range      # [BLOCK_M]
    q_mask = q_row < S                      # [BLOCK_M]

    # Load Q block: [BLOCK_M, HEAD_DIM]
    q_off = q_row[:, None] * stride_qs + d_range[None, :] * stride_qd
    Q     = tl.load(Q_base + q_off, mask=q_mask[:, None], other=0.0)

    # ── Flash Attention 2 online-softmax loop ──────────────────────────────
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M],              dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM],    dtype=tl.float32)

    # Pre-scale Q by 1/sqrt(D) so we avoid a multiplication inside the loop.
    # scale (a Python float) is exact for D=64: scale=1/8=0.125
    if USE_FP16:
        Q_c = (Q * scale).to(Q.dtype)   # pre-scale in native dtype
    else:
        Q_c = Q.to(tl.float32) * scale  # pre-scale in fp32

    num_kv_blocks = tl.cdiv(S, BLOCK_N)
    for j in range(num_kv_blocks):
        kv_row  = j * BLOCK_N + n_range
        kv_mask = kv_row < S

        # Load K block: [BLOCK_N, HEAD_DIM]
        k_off = kv_row[:, None] * stride_ks + d_range[None, :] * stride_kd
        K = tl.load(K_base + k_off, mask=kv_mask[:, None], other=0.0,
                    eviction_policy="evict_last")

        # QK^T → [BLOCK_M, BLOCK_N]   (scale already applied to Q)
        if USE_FP16:
            # Accumulate into fp32 directly from native bf16/fp16 tensor cores
            S_ij = tl.dot(Q_c, tl.trans(K), out_dtype=tl.float32)
        else:
            # tf32 tensor cores for fp32
            S_ij = tl.dot(Q_c, tl.trans(K.to(tl.float32)), allow_tf32=True)

        # Attention mask: [BLOCK_M, BLOCK_N]
        mask_off = (off_b * stride_mb + 0 * stride_mh
                    + q_row[:, None]  * stride_ms
                    + kv_row[None, :] * stride_mn)
        mask_val = tl.load(
            Mask_ptr + mask_off,
            mask=q_mask[:, None] & kv_mask[None, :],
            other=0.0,
        )
        S_ij = S_ij + mask_val.to(tl.float32)

        # Mask padding positions
        S_ij = tl.where(kv_mask[None, :], S_ij, float('-inf'))

        # Online softmax update (Flash-Attention 2 algorithm)
        m_ij  = tl.max(S_ij, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i  - m_new)
        P     = tl.exp(S_ij - m_new[:, None])     # [BLOCK_M, BLOCK_N]

        # Load V block: [BLOCK_N, HEAD_DIM]
        v_off = kv_row[:, None] * stride_vs + d_range[None, :] * stride_vd
        V = tl.load(V_base + v_off, mask=kv_mask[:, None], other=0.0,
                    eviction_policy="evict_first")

        # P @ V – accumulate into fp32 via native dtype tensor cores
        if USE_FP16:
            pv = tl.dot(P.to(V.dtype), V, out_dtype=tl.float32)
        else:
            # V in fp32; tf32 accumulation
            pv = tl.dot(P, V.to(tl.float32), allow_tf32=True)

        acc = acc * alpha[:, None] + pv
        l_i = l_i * alpha + tl.sum(P, axis=1)
        m_i = m_new

    # Normalize and write output in [B, S, H, D] layout
    acc  = acc / l_i[:, None]
    o_off = q_row[:, None] * stride_os + d_range[None, :] * stride_od
    tl.store(O_base + o_off, acc.to(Q.dtype), mask=q_mask[:, None])


@torch.fx.wrap
def triton_sdpa_transpose(query, key, value, attn_mask):
    """
    Fused scaled_dot_product_attention + transpose(1,2).

    Inputs:
      query, key, value : [B, H, S, D]  (may be non-contiguous)
      attn_mask         : [B, 1, S, S]

    Output:
      [B, S, H, D]   (i.e. sdpa_result.transpose(1, 2))
    """
    B  = query.shape[0]
    H  = query.shape[1]
    S  = query.shape[2]
    D  = query.shape[3]

    scale = float(D) ** -0.5

    # Use fp16 tensor-core path for half-precision inputs
    use_fp16 = query.dtype in (torch.float16, torch.bfloat16)

    # Output layout: [B, S, H, D]  (contiguous allocation)
    out = torch.empty((B, S, H, D), dtype=query.dtype, device=query.device)

    grid = lambda meta: (B * H, triton.cdiv(S, meta['BLOCK_M']))

    flash_attn_fwd_kernel[grid](
        query, key, value, attn_mask, out,
        query.stride(0),    query.stride(1),    query.stride(2),    query.stride(3),
        key.stride(0),      key.stride(1),      key.stride(2),      key.stride(3),
        value.stride(0),    value.stride(1),    value.stride(2),    value.stride(3),
        attn_mask.stride(0), attn_mask.stride(1), attn_mask.stride(2), attn_mask.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, H, S, D,
        scale,
        HEAD_DIM=64,
        USE_FP16=use_fp16,
    )

    return out