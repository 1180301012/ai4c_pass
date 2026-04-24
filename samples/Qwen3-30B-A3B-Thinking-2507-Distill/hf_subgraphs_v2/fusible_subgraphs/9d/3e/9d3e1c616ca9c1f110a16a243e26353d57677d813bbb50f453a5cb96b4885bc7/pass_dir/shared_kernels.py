"""
Shared Triton kernels and dispatch wrapper used by BOTH pass files.
Both passes import _dispatch from here so they return the SAME Python
function object, satisfying the replacement_func_limit constraint.
"""
import torch
import triton
import triton.language as tl


# ── Triton kernel: fused 1x1-conv + BN-inference + residual-add ──────────────
# Grid: (ceil(M/BLOCK_M), N_batch) where M = H*W
# Each block handles BLOCK_M spatial positions for one batch element.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'BLOCK_K': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128,'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128,'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 256,'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=8, num_stages=4),
    ],
    key=['HW', 'C_in', 'C_out'],
)
@triton.jit
def _conv1x1_bn_add_kernel(
    x_ptr, w_ptr, rm_ptr, rv_ptr, bn_w_ptr, bn_b_ptr, res_ptr, out_ptr,
    M, HW, C_in, C_out, N_batch,
    eps,
    stride_xn, stride_xc,   # x strides: NCHW  (stride_xc = HW)
    stride_wm, stride_wk,   # w strides: [C_out, C_in, 1, 1]
    stride_res_n, stride_res_c,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Each program handles a tile [BLOCK_M, BLOCK_N] of the output.
    pid(0) = m-block index, pid(1) = batch index n.
    """
    m_pid  = tl.program_id(0)
    n_pid  = tl.program_id(1)   # batch index in [0, N_batch)

    m_start = m_pid * BLOCK_M
    m_offs  = m_start + tl.arange(0, BLOCK_M)
    m_mask  = m_offs < M

    n       = n_pid
    m_base  = n * HW                 # offset to this batch's spatial data

    # ── GEMM: x [M, C_in] @ w^T [C_in, C_out] ──────────────────────────────
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, C_in, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < C_in

        # x tile [BLOCK_M, BLOCK_K]  –  NCHW: x[n, ci, hw] = x_ptr[n*C_in*HW + ci*HW + hw]
        x_offs = m_base + k_offs[:, None] * HW + m_offs[None, :]
        x_tile = tl.load(x_ptr + x_offs,
                         mask=m_mask[None, :] & k_mask[:, None],
                         other=0.0)

        # w tile [BLOCK_N, BLOCK_K]  –  w[c_out, c_in]: stride_wm=C_in, stride_wk=1
        w_offs = n_offs[:, None] * stride_wm + k_offs[None, :] * stride_wk
        w_tile = tl.load(w_ptr + w_offs,
                         mask=n_mask[:, None] & k_mask[None, :],
                         other=0.0)

        acc = tl.dot(x_tile, tl.trans(w_tile), acc)

    # ── BN inference + residual add ──────────────────────────────────────────
    n_offs = tl.arange(0, BLOCK_N)
    n_mask = n_offs < C_out

    rm    = tl.load(rm_ptr   + n_offs, mask=n_mask).to(tl.float32)
    rv    = tl.load(rv_ptr   + n_offs, mask=n_mask).to(tl.float32)
    bn_w  = tl.load(bn_w_ptr + n_offs, mask=n_mask).to(tl.float32)
    bn_b  = tl.load(bn_b_ptr + n_offs, mask=n_mask).to(tl.float32)
    inv_std = 1.0 / tl.sqrt(rv + eps)

    res_offs = n * C_out * HW + n_offs[:, None] * HW + m_offs[None, :]
    res = tl.load(res_ptr + res_offs,
                  mask=m_mask[None, :] & n_mask[:, None],
                  other=0.0).to(tl.float32)

    acc_f32  = acc.to(res.dtype.element_ty)
    bn_out   = (acc_f32 - rm[None, :]) * inv_std[None, :] * bn_w[None, :] + bn_b[None, :]
    out_val  = res + bn_out

    tl.store(out_ptr + res_offs, out_val, mask=m_mask[None, :] & n_mask[:, None])


# ── Python wrapper (called by the dispatch below) ────────────────────────────
def _fused_impl(x, conv_w, running_mean, running_var, bn_bias, bn_weight, residual, route):
    """
    Computes: out = BN_inference(conv1x1(x, conv_w)) + residual

    Arguments follow the order expected by the dispatch:
      route="deeppose": (x, conv_w, running_mean, running_var, bn_weight, bn_bias, residual)
      route="resnet10t": (x, conv_w, running_mean, running_var, bn_bias, bn_weight, residual)
    """
    x = x.float()
    residual = residual.float()
    bn_bias  = bn_bias.float()
    bn_weight = bn_weight.float()
    conv_w   = conv_w.float()

    N, C_in, H, W = x.shape
    C_out    = conv_w.shape[0]
    HW       = H * W
    M        = N * HW

    # Unify BN weight/bias order: always (running_mean, running_var, weight, bias)
    # deeppose order: (rm=bn_w, rv, bias=bn_b, weight) → (rm, rv, weight, bias)
    # resnet10t order: (rm, rv, bias, weight)        → (rm, rv, weight, bias)
    if route == "deeppose":
        rm, rv, bw, bb = running_mean, running_var, bn_weight, bn_bias
    else:
        rm, rv, bw, bb = running_mean, running_var, bn_weight, bn_bias

    out = torch.empty_like(residual)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), N)
    _conv1x1_bn_add_kernel[grid](
        x, conv_w, rm, rv, bw, bb, residual, out,
        M, HW, C_in, C_out, N,
        1e-5,
        x.stride(0), x.stride(1),   # stride_xn, stride_xc
        conv_w.stride(0), conv_w.stride(1),  # stride_wm, stride_wk
        residual.stride(0), residual.stride(1),  # stride_res_n, stride_res_c
    )
    return out


# ── Shared dispatch wrapper returned by BOTH pass files ──────────────────────
@torch.fx.wrap
def _dispatch(a0, a1, a2, a3, a4, a5, a6, route):
    """
    Unified signature shared by both pass files.

    deeppose (conv_w=a0, rm=a1, rv=a2, bn_w=a3, bn_b=a4, x=a5, res=a6):
        pattern: conv2d(in_6, in_4, …) → BN(conv2d, in_0, in_1, in_3, in_2, …) → BN_out += in_5
        dispatch args: (in_4, in_0, in_1, in_3, in_2, in_6, in_5, "deeppose")
        → (conv_w, rm, rv, bn_w, bn_b, x, res)

    resnet10t (conv_w=a0, rm=a1, rv=a2, bn_bias=a3, bn_weight=a4, x=a5, res=a6):
        pattern: conv2d(in_5, in_0, …) → BN(conv2d, in_1, in_2, in_4, in_3, …) → in_6 += BN_out
        dispatch args: (in_0, in_1, in_2, in_4, in_3, in_5, in_6, "resnet10t")
        → (conv_w, rm, rv, bn_bias, bn_weight, x, res)
    """
    if route == "deeppose":
        return _fused_impl(a0, a1, a2, a3, a4, a5, a6, route)
    else:
        # resnet10t: bn_weight=a3, bn_bias=a2 already swapped from pattern order
        # so pass rm=a1, rv=a2, bn_bias=a3, bn_weight=a4
        return _fused_impl(a0, a1, a2, a3, a4, a5, a6, route)