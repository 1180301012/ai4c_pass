"""
Fused QKV GEMM+rearrange for convit_small (H=9, T=197, D=48).

One Triton kernel replaces linear+reshape+permute+unbind+transpose:
  Grid = (ceil(T/BLOCK_M), H): each block computes Q+K+V for one head.
  Beats cuBLAS for small T by fusing matmul and rearrangement.
"""

import torch
import triton
import triton.language as tl

# ── Constants ─────────────────────────────────────────────────────────────────
_H     = 9
_T     = 197
_D     = 48
_D_PAD = 64   # next power-of-2 >= 48

_dtype_tl = {
    torch.float32:  tl.float32,
    torch.float16:  tl.float16,
    torch.bfloat16: tl.bfloat16,
}

# ── Fused GEMM + rearrange kernel ─────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_K': 16, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_K': 64, 'num_stages': 2, 'num_warps': 4}),
    ],
    key=['K'],
)
@triton.jit
def _qkv_rearrange_9h(
    x_ptr, w_ptr,           # [T, K]  and  [3*H*D, K]
    q_ptr, kt_ptr, v_ptr,   # Q[H,T,D], K_T[H,D,T], V[H,T,D]
    T, K,
    H:      tl.constexpr,
    D:      tl.constexpr,
    D_PAD:  tl.constexpr,
    DTYPE:  tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    t_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    t_mask = t_offs < T
    d_offs = tl.arange(0, D_PAD)
    d_mask = d_offs < D
    k_offs = tl.arange(0, BLOCK_K)

    q_base = 0 * H * D + pid_h * D
    k_base = 1 * H * D + pid_h * D
    v_base = 2 * H * D + pid_h * D

    acc_q = tl.zeros((BLOCK_M, D_PAD), dtype=tl.float32)
    acc_k = tl.zeros((BLOCK_M, D_PAD), dtype=tl.float32)
    acc_v = tl.zeros((BLOCK_M, D_PAD), dtype=tl.float32)

    # x[t, k] → [BLOCK_M, BLOCK_K]: coalesced along k ✓
    x_ptrs  = x_ptr + t_offs[:, None] * K + k_offs[None, :]
    # w[d, k] → [D_PAD, BLOCK_K]: coalesced along k ✓
    wq_ptrs = w_ptr + (q_base + d_offs[:, None]) * K + k_offs[None, :]
    wk_ptrs = w_ptr + (k_base + d_offs[:, None]) * K + k_offs[None, :]
    wv_ptrs = w_ptr + (v_base + d_offs[:, None]) * K + k_offs[None, :]

    for k_start in range(0, K, BLOCK_K):
        k_rem  = K - k_start
        x_mask = t_mask[:, None] & (k_offs[None, :] < k_rem)
        w_mask = d_mask[:, None] & (k_offs[None, :] < k_rem)

        xt  = tl.load(x_ptrs,  mask=x_mask, other=0.0)
        wq  = tl.load(wq_ptrs, mask=w_mask, other=0.0)
        wk  = tl.load(wk_ptrs, mask=w_mask, other=0.0)
        wv  = tl.load(wv_ptrs, mask=w_mask, other=0.0)

        acc_q = tl.dot(xt, tl.trans(wq), acc_q)
        acc_k = tl.dot(xt, tl.trans(wk), acc_k)
        acc_v = tl.dot(xt, tl.trans(wv), acc_v)

        x_ptrs  += BLOCK_K
        wq_ptrs += BLOCK_K
        wk_ptrs += BLOCK_K
        wv_ptrs += BLOCK_K

    mask2d = t_mask[:, None] & d_mask[None, :]

    tl.store(q_ptr  + pid_h * T * D + t_offs[:, None] * D + d_offs[None, :],
             acc_q.to(DTYPE), mask=mask2d)
    tl.store(kt_ptr + pid_h * D * T + d_offs[None, :] * T + t_offs[:, None],
             acc_k.to(DTYPE), mask=mask2d)
    tl.store(v_ptr  + pid_h * T * D + t_offs[:, None] * D + d_offs[None, :],
             acc_v.to(DTYPE), mask=mask2d)


@torch.fx.wrap
def _qkv_compute_9heads(in_0, in_1):
    H, T, D = _H, _T, _D
    device  = in_1.device
    dtype   = in_1.dtype

    w = in_0.to(device=device, dtype=dtype)
    x = in_1.reshape(T, -1)
    K = x.shape[1]

    q   = torch.empty(H, T, D, device=device, dtype=dtype)
    k_t = torch.empty(H, D, T, device=device, dtype=dtype)
    v   = torch.empty(H, T, D, device=device, dtype=dtype)

    DTYPE = _dtype_tl[dtype]
    grid  = lambda META: (triton.cdiv(T, META['BLOCK_M']), H)

    _qkv_rearrange_9h[grid](
        x, w, q, k_t, v,
        T=T, K=K, H=H, D=D, D_PAD=_D_PAD, DTYPE=DTYPE,
    )

    return q.unsqueeze(0), k_t.unsqueeze(0), v.unsqueeze(0)


def _qkv_fused_9heads(in_0, in_1):
    result = _qkv_compute_9heads(in_0, in_1)
    return result[0], result[1], result[2]


def pattern(in_0, in_1):
    linear  = torch.nn.functional.linear(in_1, in_0, None)
    tmp_2   = linear.reshape(1, 197, 3, 9, 48)
    tmp_3   = tmp_2.permute(2, 0, 3, 1, 4)
    unbind  = tmp_3.unbind(0)
    tmp_5   = unbind[0]
    tmp_6   = unbind[1]
    tmp_7   = unbind[2]
    tmp_8   = tmp_6.transpose(-2, -1)
    return (tmp_5, tmp_8, tmp_7)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return _qkv_fused_9heads