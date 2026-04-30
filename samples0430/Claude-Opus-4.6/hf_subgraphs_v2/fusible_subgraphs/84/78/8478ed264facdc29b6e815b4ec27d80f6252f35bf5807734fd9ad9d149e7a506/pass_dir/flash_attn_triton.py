import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16}, num_warps=2, num_stages=1),
    ],
    key=['N_Q', 'N_KV', 'BLOCK_D'],
)
@triton.jit
def flash_attn_fwd_kernel(
    Q, K, V, Out,
    sm_scale,
    H, D, N_Q, N_KV,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kd, stride_kn,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_om, stride_oh, stride_od,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh % H

    # Base pointers for this batch and head
    q_base = Q + b * stride_qb + h * stride_qh
    k_base = K + b * stride_kb + h * stride_kh
    v_base = V + b * stride_vb + h * stride_vh

    # Query block offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    # Load Q block [BLOCK_M, BLOCK_D]
    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q_mask = (offs_m[:, None] < N_Q) & (offs_d[None, :] < D)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Initialize online softmax state
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # Iterate over KV blocks
    for start_n in range(0, N_KV, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # Load K block [BLOCK_D, BLOCK_N] - K layout is [B, H, D, N_KV]
        k_ptrs = k_base + offs_d[:, None] * stride_kd + offs_n[None, :] * stride_kn
        k_mask = (offs_d[:, None] < D) & (offs_n[None, :] < N_KV)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # Compute QK^T [BLOCK_M, BLOCK_N]
        qk = tl.dot(q, k)
        qk = qk * sm_scale
        # Mask invalid KV positions
        qk = tl.where(offs_n[None, :] < N_KV, qk, float('-inf'))

        # Online softmax update
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)

        # Load V block [BLOCK_N, BLOCK_D] - V layout is [B, H, N_KV, D]
        v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v_mask = (offs_n[:, None] < N_KV) & (offs_d[None, :] < D)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        # Update accumulator
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)

        m_i = m_new

    # Normalize
    acc = acc / l_i[:, None]

    # Store output in [B, N_Q, H, D] layout (permuted)
    out_base = Out + b * stride_ob + h * stride_oh
    out_ptrs = out_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    out_mask = (offs_m[:, None] < N_Q) & (offs_d[None, :] < D)
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=out_mask)


@torch.fx.wrap
def flash_attn_wrapper(Q, K, V, route):
    B, H, N_Q, D = Q.shape
    N_KV = V.shape[2]

    # Determine scale from route string
    if route == "scale_5656":
        sm_scale = 1.0 / 5.656854249492381
    elif route == "scale_8":
        sm_scale = 1.0 / 8.0
    elif route == "scale_6_d0":
        sm_scale = 1.0 / 6.0
    elif route == "scale_6_d01":
        sm_scale = 1.0 / 6.0
    elif route == "scale_6928_d01":
        sm_scale = 1.0 / 6.928203230275509
    else:
        sm_scale = 1.0 / 8.0

    # Compute BLOCK_D (next power of 2 >= D, minimum 16)
    if D <= 16:
        BLOCK_D = 16
    elif D <= 32:
        BLOCK_D = 32
    elif D <= 64:
        BLOCK_D = 64
    else:
        BLOCK_D = 128

    # Allocate output in [B, N_Q, H, D] format (result of permute(0,2,1,3) + contiguous)
    out = torch.empty((B, N_Q, H, D), dtype=Q.dtype, device=Q.device)

    # Grid: one program per query block per batch*head
    grid = lambda META: ((N_Q + META['BLOCK_M'] - 1) // META['BLOCK_M'], B * H)

    # Launch kernel
    flash_attn_fwd_kernel[grid](
        Q, K, V, out,
        sm_scale,
        H, D, N_Q, N_KV,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_D=BLOCK_D,
    )

    return out