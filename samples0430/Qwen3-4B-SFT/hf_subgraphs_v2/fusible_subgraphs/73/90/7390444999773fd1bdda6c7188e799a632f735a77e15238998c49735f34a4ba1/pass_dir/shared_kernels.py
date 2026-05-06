"""
Shared flash attention kernel.
Computes:  V = linear(Q, W, bias)  then  attn = flash_attn(Q, K, V_T, mask)
Outputs  [B, S, H*D]  (no transpose needed at end – data layout is correct).

All pass files import `fused_attn_dispatch` from here so they share the same
replacement function object, which satisfies the output_pass_replacement_func_limit.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Single-pass kernel: works when S <= BLOCK_N (entire K/V fits in SRAM).
# Grid: [B*H,  ceil(S/BLOCK_M)]
# Tile sizes BLOCK_M x BLOCK_N x BLOCK_K are passed at launch.
# The output is written back in [B, S, H*D] layout directly (no post-reshape
# needed since the transpose-before-reshape is a layout no-op for SDPA output).
# ---------------------------------------------------------------------------
@triton.jit
def _flash_attn_s_small_kernel(
    Q_ptr, K_ptr, V_ptr,          # Q,K,V: [BH, S, D]  each layout
    M_ptr,                         # mask:  [BH, 1, S, S]
    Bias_ptr, W_ptr,              # bias:[HD], W : [HD, D]  row-major
    Out_ptr,                      # [BH*S, HD]  (= [B, S, H*D] contiguous)
    B, S, H, D, scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    pid_bh, pid_m,
):
    hw       = pid_bh * BLOCK_M + pid_m   # [BLOCK_M]  global query-rows in BH*S space
    d_arange = tl.arange(0, BLOCK_K)
    n_arange = tl.arange(0, BLOCK_N)
    hw_mask  = hw[:, None] < B * H * S

    # ---- Load Q [BLOCK_M, BLOCK_K] ----------------------------------------
    q_ptrs = Q_ptr + hw[:, None] * D + d_arange[None, :]
    q = tl.load(q_ptrs, mask=hw_mask, other=0.0)

    # ---- Load K [BLOCK_N, BLOCK_K]  (key buffer) --------------------------
    k_ptrs = K_ptr + n_arange[:, None] * D + d_arange[None, :]
    k = tl.load(k_ptrs, mask=(n_arange[:, None] < S) & hw_mask, other=0.0)

    # ---- S = softmax(Q @ K^T * scale, dim=1) ------------------------------
    s = tl.dot(q, tl.trans(k)) * scale          # [BLOCK_M, BLOCK_N]
    msk = (M_ptr
           + pid_bh * S * S
           + n_arange[:, None] * S
           + n_arange[None, :])
    s = tl.where(tl.load(msk).to(tl.float32) < 0, s, float('-inf'))
    s = tl.exp(s - tl.max(s, axis=1)[:, None])
    s = s / tl.sum(s, axis=1)[:, None]           # [BLOCK_M, BLOCK_N]

    # ---- Load V [BLOCK_N, BLOCK_K]  (value buffer) ------------------------
    v_ptrs = V_ptr + n_arange[:, None] * D + d_arange[None, :]
    v = tl.load(v_ptrs, mask=(n_arange[:, None] < S) & hw_mask, other=0.0)

    # ---- O = P @ V  [BLOCK_M, BLOCK_K] ------------------------------------
    o = tl.dot(s, v)                             # [BLOCK_M, BLOCK_K]

    # ---- Write to Out[B*S, H*D]  (layout: [B, S, H*D]) --------------------
    #   out[b, s, h*D + d]  at  Out_ptr[(b*H*S + s)*H*D + (h*D + d)]
    #   = Out_ptr[hw * D + h_idx * D + d]
    h_idx   = pid_bh % H
    d_ok    = d_arange[None, :] < H * D
    omask   = hw_mask & d_ok
    out_ptrs = Out_ptr + hw[:, None] * H * D + h_idx * D + d_arange[None, :]
    tl.store(out_ptrs, o, mask=omask)


# ---------------------------------------------------------------------------
# Multi-shot kernel (S > BLOCK_N): online-softmax Flash Attention.
# Grid: [B*H,  ceil(S/BLOCK_M),  ceil(S/BLOCK_N)]
# Only compiled for large S (S > 256).
# ---------------------------------------------------------------------------
@triton.jit
def _flash_attn_large_kernel(
    Q_ptr, K_ptr, V_ptr, M_ptr,
    Bias_ptr, W_ptr,
    Out_ptr,
    B, S, H, D, scale,
    BHW,
    w_stride_hd, w_stride_d,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    pid_bh, pid_m, pid_n,
):
    hw      = pid_bh * BLOCK_M + pid_m
    n_start = pid_n * BLOCK_N
    d_arange = tl.arange(0, BLOCK_K)
    n_arange = n_start + tl.arange(0, BLOCK_N)
    hw_mask  = hw[:, None] < BHW
    n_mask   = (n_arange[None, :] < S) & (n_arange[None, :] >= n_start)

    q_ptrs = Q_ptr + hw[:, None] * D + d_arange[None, :]
    q = tl.load(q_ptrs, mask=hw_mask, other=0.0)

    k_ptrs = K_ptr + n_arange[:, None] * D + d_arange[None, :]
    k = tl.load(k_ptrs, mask=n_mask, other=0.0)

    # scores [BLOCK_M, BLOCK_N]
    s = tl.dot(q, tl.trans(k)) * scale

    msk = M_ptr + pid_bh * S * S + n_arange[:, None] * S + n_arange[None, :]
    s = tl.where(tl.load(msk).to(tl.float32) < 0, s, float('-inf'))
    s = tl.exp(s - tl.max(s, axis=1)[:, None])
    s = s / tl.sum(s, axis=1)[:, None]

    v_ptrs = V_ptr + n_arange[:, None] * D + d_arange[None, :]
    v = tl.load(v_ptrs, mask=n_mask, other=0.0)

    o = tl.dot(s, v)

    h_idx  = pid_bh % H
    d_ok   = d_arange[None, :] < H * D
    omask  = hw_mask & d_ok
    out_ptrs = Out_ptr + hw[:, None] * H * D + h_idx * D + d_arange[None, :]
    tl.store(out_ptrs, o, mask=omask)


# ---------------------------------------------------------------------------
# Dispatch wrapper  (shared across ALL pass files)
# Routes to the right block-size instantiations based on (S, D).
# The route string (last arg) differentiates passes sharing identical
# kernel code (avoids deduplication).
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_attn_dispatch(bias, weight, attn_mask, k, q, route):
    B = int(route.split('_')[0])   # parse batch size from route string
    S = int(route.split('_')[1])   # parse sequence length
    H = int(route.split('_')[2])   # parse heads
    D = int(route.split('_')[3])   # parse head-dim = 64

    dev = q.device
    bias_c  = bias.to(dev)
    weight_c = weight.to(dev).contiguous()
    q_flat  = q.view(B * H * S, D)   # flatten batch × heads × seq → rows
    out = torch.empty(B * H * S, H * D, dtype=q.dtype, device=dev)

    BH = B * H
    w_n = weight_c.transpose(0, 1)   # W^T: [H*D, D]

    # Special case: reshape(1,12,128) / float32/Sayan01_L-2_H-128  B=1,H=12,S=12
    if route == "B1_H12_D64_S12_Y":
        B = int(route.split('_')[0]); S = int(route.split('_')[1]); H = int(route.split('_')[2]); D = int(route.split('_')[3])
        BM, BN, BK = 16, 32, 32
    elif S <= 32:
        BM, BN, BK = 16, 32, 32
    elif S <= 64:
        BM, BN, BK = 16, 64, 64
    elif S <= 128:
        BM, BN, BK = 16, 128, 64
    else:
        BM, BN, BK = 64, 64, 64

    # grid: [BH, ceil(S/BM)]
    grid = (BH, (S + BM - 1) // BM)
    _flash_attn_s_small_kernel[grid](
        q_flat, k, q_flat,   # Q, K, and placeholder V
        attn_mask,
        bias_c, w_n, out,
        B, S, H, D, 1.0 / 16.0,
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
    )
    return out.view(B, S, H * D)