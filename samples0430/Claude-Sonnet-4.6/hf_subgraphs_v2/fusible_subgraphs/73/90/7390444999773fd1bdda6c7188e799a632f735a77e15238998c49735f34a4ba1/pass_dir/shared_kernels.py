"""
Shared Triton kernels for value-projection + Flash Attention fusion.

Pattern replaced:
  linear(in_3, in_1, in_0)              -> Y [B*S, H]
  view(B, -1, NH, DH)                   -> V [B, S, NH, DH]
  transpose(1, 2)                        -> V_t [B, NH, S, DH]
  [optional .to(dtype)]
  scaled_dot_product_attention(Q, K, V_t, mask)
  transpose(1, 2)                        -> out [B, S, NH, DH]
  reshape(B, S, H)                       -> out [B, S, H]

Optimization:
  - Triton GEMM for the linear projection (V stays in [B*S, H] = [B, S, NH, DH])
  - Flash Attention 2 that reads V from [B, S, NH, DH] layout (avoiding the V transpose)
  - Output written directly as [B, S, H] (avoiding the post-SDPA transpose+reshape)
"""

import torch
import triton
import triton.language as tl
import math

###############################################################################
# GEMM + Bias: Y[M, N] = X[M, K] @ W[N, K].T + b[N]
###############################################################################

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _gemm_bias_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Grouped swizzle for L2 cache
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # X ptrs: [BM, BK], W.T ptrs: [BK, BN]
    x_ptrs   = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    wt_ptrs  = W_ptr + offs_k[:, None] * stride_wk  + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_off = k * BLOCK_K
        x  = tl.load(x_ptrs,  mask=(offs_m[:, None] < M) & (offs_k[None, :] + k_off < K), other=0.0)
        wt = tl.load(wt_ptrs, mask=(offs_k[:, None] + k_off < K) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(x, wt, acc)
        x_ptrs  += BLOCK_K * stride_xk
        wt_ptrs += BLOCK_K * stride_wk

    if HAS_BIAS:
        b = tl.load(B_ptr + offs_n, mask=offs_n < N)
        acc = acc + b[None, :]

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, acc.to(Y_ptr.dtype.element_ty), mask=out_mask)


###############################################################################
# Flash Attention 2 Forward Pass
#
# Q, K : [B, NH, S, DH]   (contiguous standard layout)
# V    : [B, S,  NH, DH]  (comes from linear output – no transpose needed)
# Mask : [B, 1,  S,  S ]  (additive attention mask)
# Out  : [B, S,  H  ]     where H = NH * DH
#
# Grid : (B * NH,  cdiv(S, BLOCK_M))
###############################################################################

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128,'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128},num_warps=8, num_stages=2),
    ],
    key=['NH', 'S', 'HEAD_DIM'],
)
@triton.jit
def _flash_attn_fwd(
    Q_ptr, K_ptr, V_ptr, Mask_ptr, Out_ptr,
    B, NH, S,
    sm_scale,
    # Q, K strides: [B, NH, S, DH]
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    # V strides: [B, S, NH, DH]
    stride_vb, stride_vs, stride_vh, stride_vd,
    # Mask strides: [B, 1, S, S]  (head dim is broadcast)
    stride_mb, stride_mm, stride_mn,
    # Out strides: [B, S, H] where H = NH * DH
    stride_ob, stride_os,
    HAS_MASK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,  # compile-time constant = 64
):
    bh_idx  = tl.program_id(0)
    m_block = tl.program_id(1)

    b_idx = bh_idx // NH
    h_idx = bh_idx % NH

    offs_m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    # Load Q tile: [BLOCK_M, HEAD_DIM]
    q_ptrs = (Q_ptr
              + b_idx * stride_qb
              + h_idx * stride_qh
              + offs_m[:, None] * stride_qs
              + offs_d[None, :] * stride_qd)
    q_mask = offs_m[:, None] < S
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Running online-softmax stats
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    k_base = K_ptr + b_idx * stride_kb + h_idx * stride_kh
    v_base = V_ptr + b_idx * stride_vb + h_idx * stride_vh

    for start_n in range(0, S, BLOCK_N):
        offs_n  = start_n + tl.arange(0, BLOCK_N)
        n_mask  = offs_n < S

        # Load K tile: [HEAD_DIM, BLOCK_N]  (transposed for QK product)
        k_ptrs = (k_base
                  + offs_n[None, :] * stride_ks
                  + offs_d[:, None] * stride_kd)
        k = tl.load(k_ptrs, mask=n_mask[None, :], other=0.0)

        # QK: [BLOCK_M, BLOCK_N]
        qk = tl.dot(q, k) * sm_scale

        if HAS_MASK:
            m_ptrs = (Mask_ptr
                      + b_idx * stride_mb
                      + offs_m[:, None] * stride_mm
                      + offs_n[None, :] * stride_mn)
            mask_vals = tl.load(m_ptrs,
                                mask=q_mask & n_mask[None, :],
                                other=0.0)
            qk = qk + mask_vals

        # Mask out-of-bounds positions to -inf
        qk = tl.where(n_mask[None, :], qk, float("-inf"))

        # Online softmax update
        row_max   = tl.max(qk, axis=1)
        m_i_new   = tl.maximum(m_i, row_max)
        alpha     = tl.exp(m_i - m_i_new)
        p         = tl.exp(qk - m_i_new[:, None])

        # Load V tile: V is [B, S, NH, DH] → access V[b, n, h, :]
        v_ptrs = (v_base
                  + offs_n[:, None] * stride_vs
                  + offs_d[None, :] * stride_vd)
        v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(q.dtype), v)
        m_i = m_i_new

    # Normalize
    acc = acc / l_i[:, None]

    # Write output: Out[b, m, h*DH + d]
    # stride_os = NH * DH  (= H)
    out_ptrs = (Out_ptr
                + b_idx * stride_ob
                + offs_m[:, None] * stride_os
                + h_idx * HEAD_DIM
                + offs_d[None, :])
    tl.store(out_ptrs, acc.to(q.dtype), mask=q_mask)


###############################################################################
# Wrapper: fused linear + flash-attention
###############################################################################

@torch.fx.wrap
def _dispatch(in_0, in_1, in_2, in_3, in_4, in_5, route):
    """
    in_0 : bias   [H]
    in_1 : weight [H, H]
    in_2 : mask   [B, 1, S, S]   (additive, on GPU)
    in_3 : hidden [B, S, H]      contiguous
    in_4 : key    [B, NH, S, DH]
    in_5 : query  [B, NH, S, DH]
    route: identifies the shape variant (ignored at kernel level)

    All tensor operations must go through Triton kernels.
    Only torch.empty / torch.empty_like / etc. are allowed as torch APIs.
    """
    q = in_5
    B  = q.shape[0]
    NH = q.shape[1]
    S  = q.shape[2]
    DH = q.shape[3]   # always 64
    H  = NH * DH
    M  = B * S

    device = in_3.device
    dtype  = in_3.dtype

    # ---------- GEMM: V_flat = in_3 @ in_1.T + in_0 ----------
    # in_3 is [B, S, H] contiguous – treat as [M=B*S, H] via strides:
    #   stride_xm = in_3.stride(1)  (= H, step from one seq-pos to next)
    #   stride_xk = in_3.stride(2)  (= 1)
    # in_1 is [H, H] (weight), in_0 is [H] (bias)
    v_flat = torch.empty((M, H), device=device, dtype=dtype)

    grid_gemm = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(H, meta['BLOCK_N']),
    )
    _gemm_bias_kernel[grid_gemm](
        in_3, in_1, in_0, v_flat,
        M, H, H,
        in_3.stride(1), in_3.stride(2),   # stride_xm, stride_xk
        in_1.stride(0), in_1.stride(1),   # stride_wn, stride_wk
        v_flat.stride(0), v_flat.stride(1),
        HAS_BIAS=True,
    )

    # ---------- Flash Attention ----------
    # v_flat is [B*S, H] in memory; interpret as [B, S, NH, DH] using strides:
    #   V[b, s, h, d] = v_flat_ptr + b*(S*H) + s*H + h*DH + d
    stride_vb = S * H   # advance one batch
    stride_vs = H       # advance one seq position  (= v_flat.stride(0))
    stride_vh = DH      # advance one head
    stride_vd = 1       # advance within head dim

    out = torch.empty((B, S, H), device=device, dtype=dtype)
    sm_scale = 1.0 / math.sqrt(DH)
    has_mask = in_2 is not None

    grid_attn = lambda meta: (
        B * NH,
        triton.cdiv(S, meta['BLOCK_M']),
    )
    _flash_attn_fwd[grid_attn](
        q, in_4, v_flat,
        in_2 if has_mask else q,   # dummy ptr when no mask
        out,
        B, NH, S,
        sm_scale,
        q.stride(0),    q.stride(1),    q.stride(2),    q.stride(3),
        in_4.stride(0), in_4.stride(1), in_4.stride(2), in_4.stride(3),
        stride_vb, stride_vs, stride_vh, stride_vd,
        in_2.stride(0) if has_mask else 0,
        in_2.stride(2) if has_mask else 0,
        in_2.stride(3) if has_mask else 0,
        out.stride(0), out.stride(1),
        HAS_MASK=has_mask,
        HEAD_DIM=DH,
    )
    return out