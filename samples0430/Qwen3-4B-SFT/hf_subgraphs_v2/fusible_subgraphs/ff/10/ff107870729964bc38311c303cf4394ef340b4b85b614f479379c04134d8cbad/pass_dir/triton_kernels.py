"""
Shared Triton kernels for fused conv1x1 + batch_norm + relu + upcast/dn cast.

Two patterns are supported:
  1. Conv1x1 GEMM as output of a 1x1 conv2d, followed by bn, relu, and cast:
     out = conv1x1(x, W) + bias
          = BN_infer(out, running_mean, running_var, bn_weight, bn_bias)
          = relu(bn_out)
          = out.to(target_dtype)

  2. BN + relu + cast (elementwise post-processing):
     out = BN_infer(x, running_mean, running_var, bn_weight, bn_bias)
          = relu(bn_out)
          = out.to(target_dtype)
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Shared Triton kernel helper
# ---------------------------------------------------------------------------

def _bn_params_to_float32(running_mean, running_var, bn_weight, bn_bias):
    """Cast all BN buffers to float32 for computation."""
    return (
        running_mean.float().contiguous().view(-1),
        running_var.float().contiguous().view(-1),
        bn_weight.float().contiguous().view(-1),
        bn_bias.float().contiguous().view(-1),
    )


# DTYPE SYMBOLS
_DTYPE_FP16  = tl.float16
_DTYPE_BF16  = tl.bfloat16


# ---------------------------------------------------------------------------
# Kernel 1: Fused conv1x1 GEMM  +  BN inference  +  ReLU  →  fp16/bf16
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _conv1x1_bn_relu_fused_kernel(
    x_ptr, w_ptr, b_ptr,
    bn_mean_ptr, bn_var_ptr, bn_w_ptr, bn_b_ptr,
    out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_odm, stride_odn,
    EPS: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused 1x1-conv GEMM + BN-inference + ReLU.
    x:  [M, K]  (spatial pixels × in-channels)
    w:  [N, K]  (out-channels × in-channels, from conv weight [N,Cin,1,1])
    b:  [N]     (conv bias)
    out: [M, N]  (spatial pixels × out-channels)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers to the start of this (pid_m, pid_n) tile
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # GEMM: acc += x @ w^T  (loop over K)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        mask_k = offs_k < K - k * BLOCK_K

        xa = tl.load(x_ptrs,
                     mask=(offs_m[:, None] < M) & mask_k[None, :],
                     other=0.0).to(tl.float32)

        wb = tl.load(w_ptrs.T,
                     mask=(offs_n[:, None] < N) & mask_k[None, :],
                     other=0.0).to(tl.float32)

        acc = tl.dot(xa, wb, acc)

        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # Add conv bias
    b_val = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc = acc + b_val[None, :]

    # BN inference: normalize + affine
    bn_mean = tl.load(bn_mean_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    bn_var  = tl.load(bn_var_ptr  + offs_n, mask=offs_n < N, other=1.0).to(tl.float32)
    bn_w    = tl.load(bn_w_ptr    + offs_n, mask=offs_n < N, other=1.0).to(tl.float32)
    bn_b    = tl.load(bn_b_ptr    + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)

    bn_var_inv = tl.rsqrt(bn_var + EPS)
    scale      = bn_w * bn_var_inv          # gamma / sqrt(var + eps)
    shift      = bn_b - bn_mean * scale     # beta  - mean * scale
    acc = acc * scale[None, :] + shift[None, :]

    # ReLU
    acc = tl.maximum(acc, 0.0)

    # Store in the target dtype
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    out_ptrs = out_ptr + offs_m[:, None] * stride_odm + offs_n[None, :] * stride_odn

    if OUT_DTYPE == _DTYPE_FP16:
        tl.store(out_ptrs, acc.to(tl.float16), mask=mask_out)
    else:
        tl.store(out_ptrs, acc.to(tl.bfloat16), mask=mask_out)


# ---------------------------------------------------------------------------
# Kernel 2A: Fused BN + ReLU (elementwise) → fp16/bf16
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512},  num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_stages=2, num_warps=8),
    ],
    key=['N_ELEM'],
)
@triton.jit
def _bn_relu_cast_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, gamma_ptr, beta_ptr,
    N_ELEM,
    C,
    EPS: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Elementwise: out = relu(BN_infer(x)) with cast to output dtype.
    x is stored in NCHW, channel index = (flat_index // HW) % C.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_ELEM

    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # Channel index for NCHW layout
    channel_idx = (offs // (N_ELEM // C)) % C

    mean   = tl.load(mean_ptr   + channel_idx,  mask=mask, other=0.0).to(tl.float32)
    var    = tl.load(var_ptr    + channel_idx,  mask=mask, other=1.0).to(tl.float32)
    gamma  = tl.load(gamma_ptr  + channel_idx,  mask=mask, other=1.0).to(tl.float32)
    beta   = tl.load(beta_ptr   + channel_idx,  mask=mask, other=0.0).to(tl.float32)

    bn_var_inv = tl.rsqrt(var + EPS)
    scale = gamma * bn_var_inv
    shift = beta - mean * scale

    out = tl.maximum(x * scale + shift, 0.0)

    if OUT_DTYPE == _DTYPE_FP16:
        tl.store(out_ptr + offs, out.to(tl.float16),   mask=mask)
    else:
        tl.store(out_ptr + offs, out.to(tl.bfloat16),  mask=mask)


# ---------------------------------------------------------------------------
# Kernel 3: Full Branch 2 (conv1x1 + BN + ReLU + Cast)
#   NOTE: this covers the case where we want to fuse the PRE-conv1x1 BN+relu
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _conv1x1_conv2_bn_relu_fused_kernel(
    x_ptr, w_ptr, b_ptr,
    bn_mean_ptr, bn_var_ptr, bn_w_ptr, bn_b_ptr,
    conv2_x_ptr, conv2_w_ptr, conv2_b_ptr,
    out_ptr,
    M, N, K,
    # 1x1 conv 1 params
    stride_xm1, stride_xk1,
    stride_wn1, stride_wk1,
    # conv2d params
    stride_xm2, stride_xk2,
    stride_wn2, stride_wk2,
    stride_odm, stride_odn,
    EPS: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs_1 = x_ptr            + offs_m[:, None] * stride_xm1 + offs_k[None, :] * stride_xk1
    w_ptrs_1 = w_ptr            + offs_n[:, None] * stride_wn1 + offs_k[None, :] * stride_wk1

    x_ptrs_2 = conv2_x_ptr      + offs_m[:, None] * stride_xm2 + offs_k[None, :] * stride_xk2
    w_ptrs_2 = conv2_w_ptr      + offs_n[:, None] * stride_wn2 + offs_k[None, :] * stride_wk2

    # ---- First 1x1 conv GEMM (no bias) ----
    acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        mask_k = offs_k < K - k * BLOCK_K
        xa = tl.load(x_ptrs_1,
                     mask=(offs_m[:, None] < M) & mask_k[None, :],
                     other=0.0).to(tl.float32)
        wb = tl.load(w_ptrs_1.T,
                     mask=(offs_n[:, None] < N) & mask_k[None, :],
                     other=0.0).to(tl.float32)
        acc1 = tl.dot(xa, wb, acc1)
        x_ptrs_1 += BLOCK_K * stride_xk1
        w_ptrs_1 += BLOCK_K * stride_wk1

    # ---- Add conv2 bias ----
    cb = tl.load(conv2_b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc1 = acc1 + cb[None, :]

    # ---- BN inference ----
    bn_mean = tl.load(bn_mean_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    bn_var  = tl.load(bn_var_ptr  + offs_n, mask=offs_n < N, other=1.0).to(tl.float32)
    bn_w    = tl.load(bn_w_ptr    + offs_n, mask=offs_n < N, other=1.0).to(tl.float32)
    bn_b    = tl.load(bn_b_ptr    + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)

    bn_var_inv = tl.rsqrt(bn_var + EPS)
    scale = bn_w * bn_var_inv
    shift = bn_b - bn_mean * scale
    acc1 = acc1 * scale[None, :] + shift[None, :]

    # ---- ReLU ----
    acc1 = tl.maximum(acc1, 0.0)

    # ---- Store conv2 output in correct dtype ----
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    out_ptrs = out_ptr + offs_m[:, None] * stride_odm + offs_n[None, :] * stride_odn

    if OUT_DTYPE == _DTYPE_FP16:
        tl.store(out_ptrs, acc1.to(tl.float16), mask=mask_out)
    else:
        tl.store(out_ptrs, acc1.to(tl.bfloat16), mask=mask_out)


def _launch_conv1x1_bn_relu(
        out, x, w, b,
        bn_mean, bn_var, bn_w, bn_b,
        M, N, K,
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_odm, stride_odn,
        eps=1e-5,
):
    """Launch the fused conv1x1+BN+ReLU kernel."""
    output_dtypes = {
        torch.float16:  _DTYPE_FP16,
        torch.bfloat16: _DTYPE_BF16,
    }
    out_dtype = output_dtypes.get(out.dtype, _DTYPE_FP16)

    out_flat = out.view(M, N)
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N']),
    )
    _conv1x1_bn_relu_fused_kernel[grid](
        x, w, b,
        bn_mean, bn_var, bn_w, bn_b,
        out_flat,
        M, N, K,
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_odm, stride_odn,
        eps=eps,
        OUT_DTYPE=out_dtype,
    )
    return out


def _launch_bn_relu_cast(
        out, x,
        bn_mean, bn_var, bn_w, bn_b,
        N_ELEM, C,
        eps=1e-5,
):
    """Launch the fused BN+ReLU cast kernel."""
    output_dtypes = {
        torch.float16:  _DTYPE_FP16,
        torch.bfloat16: _DTYPE_BF16,
    }
    out_dtype = output_dtypes.get(out.dtype, _DTYPE_FP16)

    grid = lambda META: (triton.cdiv(N_ELEM, META['BLOCK_SIZE']),)
    _bn_relu_cast_kernel[grid](
        x, out,
        bn_mean, bn_var, bn_w, bn_b,
        N_ELEM,
        C,
        eps=eps,
        OUT_DTYPE=out_dtype,
    )
    return out