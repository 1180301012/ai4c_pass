import torch
import triton
import triton.language as tl
import math


def pattern(in_0, in_1, in_2, in_3, in_4, in_5, vb, vh, rb, rs, rd):
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_3 = linear.view(vb, -1, vh, 64)
    tmp_4 = tmp_3.transpose(1, 2)
    sdpa = torch.nn.functional.scaled_dot_product_attention(in_5, in_4, tmp_4, attn_mask=in_2, dropout_p=0.0, is_causal=False)
    tmp_6 = sdpa.transpose(1, 2)
    tmp_7 = tmp_6.reshape(rb, rs, rd)
    return tmp_7


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, vb, vh, rb, rs, rd):
    return (in_0, in_1, in_2, in_3, in_4, in_5)


# ============ GEMM Kernel (no autotune) ============
@triton.jit
def gemm_kernel(
    A_ptr, B_ptr, bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k
        a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        b_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    # Store result
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(C_ptr.dtype.element_ty), mask=c_mask)


# ============ Flash Attention Kernel ============
@triton.jit
def flash_attn_kernel(
    Q_ptr, K_ptr, V_ptr, Mask_ptr, Out_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vs, stride_vhd,
    stride_mb, stride_mh, stride_ms, stride_mn,
    stride_ob, stride_os, stride_ohd,
    B, H, S, D: tl.constexpr,
    scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh % H

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offs = tl.arange(0, D)

    # Load Q block: (BLOCK_M, D)
    q_ptrs = Q_ptr + b * stride_qb + h * stride_qh + m_offs[:, None] * stride_qs + d_offs[None, :] * stride_qd
    q_mask = m_offs[:, None] < S
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Online softmax accumulators
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)

    # Iterate over K/V blocks
    for n_start in range(0, S, BLOCK_N):
        n_offs = n_start + tl.arange(0, BLOCK_N)

        # Load K block: (BLOCK_N, D)
        k_ptrs = K_ptr + b * stride_kb + h * stride_kh + n_offs[:, None] * stride_ks + d_offs[None, :] * stride_kd
        k_mask = n_offs[:, None] < S
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # QK^T: (BLOCK_M, BLOCK_N) - use native dtype for tensor cores
        qk = tl.dot(q, tl.trans(k)).to(tl.float32) * scale

        # Load and apply mask
        mask_ptrs = Mask_ptr + b * stride_mb + h * stride_mh + m_offs[:, None] * stride_ms + n_offs[None, :] * stride_mn
        attn_mask = tl.load(mask_ptrs, mask=(m_offs[:, None] < S) & (n_offs[None, :] < S), other=0.0).to(tl.float32)
        qk = qk + attn_mask

        # Mask out invalid positions
        qk = tl.where((m_offs[:, None] < S) & (n_offs[None, :] < S), qk, float('-inf'))

        # Online softmax
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])

        acc = acc * alpha[:, None]
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

        # Load V block
        v_ptrs = V_ptr + b * stride_vb + n_offs[:, None] * stride_vs + (h * D + d_offs[None, :]) * stride_vhd
        v_mask = n_offs[:, None] < S
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        acc += tl.dot(p.to(v.dtype), v).to(tl.float32)

    # Normalize
    acc = acc / l_i[:, None]

    # Write output in (B, S, H*D) format
    out_ptrs = Out_ptr + b * stride_ob + m_offs[:, None] * stride_os + (h * D + d_offs[None, :]) * stride_ohd
    out_mask = m_offs[:, None] < S
    tl.store(out_ptrs, acc.to(Out_ptr.dtype.element_ty), mask=out_mask)


@torch.fx.wrap
def fused_linear_attention(in_0, in_1, in_2, in_3, in_4, in_5):
    # in_0: bias (D_out,)
    # in_1: weight (D_out, D_in)
    # in_2: attention mask (B_mask, H_mask, S, S)
    # in_3: hidden states (B, S, D_in)
    # in_4: key (B, H, S, D)
    # in_5: query (B, H, S, D)

    B = in_5.shape[0]
    H = in_5.shape[1]
    S = in_5.shape[2]
    D = in_5.shape[3]
    D_in = in_3.shape[2]
    D_out = in_1.shape[0]
    M = B * S
    HD = H * D

    # Step 1: Linear layer
    V_linear = torch.empty((M, D_out), dtype=in_3.dtype, device=in_3.device)

    BLOCK_M_G = 64
    BLOCK_N_G = 64
    BLOCK_K_G = 64
    grid_m = (M + BLOCK_M_G - 1) // BLOCK_M_G
    grid_n = (D_out + BLOCK_N_G - 1) // BLOCK_N_G

    gemm_kernel[(grid_m, grid_n)](
        in_3, in_1, in_0, V_linear,
        M, D_out, D_in,
        D_in, 1,
        1, D_in,
        D_out, 1,
        BLOCK_M=BLOCK_M_G, BLOCK_N=BLOCK_N_G, BLOCK_K=BLOCK_K_G,
        num_warps=4, num_stages=2,
    )

    # Step 2: Flash Attention
    Out = torch.empty((B, S, HD), dtype=in_3.dtype, device=in_3.device)

    scale = 1.0 / math.sqrt(D)

    # Strides for Q and K: (B, H, S, D) contiguous
    stride_qb = H * S * D
    stride_qh = S * D
    stride_qs = D
    stride_qd = 1

    # Mask broadcasting
    mask_b = in_2.shape[0]
    mask_h = in_2.shape[1]
    stride_mn = 1
    stride_ms = S
    stride_mh = S * S if mask_h > 1 else 0
    stride_mb = mask_h * S * S if mask_b > 1 else 0

    BLOCK_M_A = 64
    BLOCK_N_A = 64
    num_m_blocks = (S + BLOCK_M_A - 1) // BLOCK_M_A

    flash_attn_kernel[(B * H, num_m_blocks)](
        in_5, in_4, V_linear, in_2, Out,
        stride_qb, stride_qh, stride_qs, stride_qd,
        stride_qb, stride_qh, stride_qs, stride_qd,
        S * HD, HD, 1,
        stride_mb, stride_mh, stride_ms, stride_mn,
        S * HD, HD, 1,
        B, H, S, D,
        scale,
        BLOCK_M=BLOCK_M_A, BLOCK_N=BLOCK_N_A,
        num_warps=4, num_stages=2,
    )

    return Out


def replacement_func():
    return fused_linear_attention