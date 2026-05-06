import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────
#  Helper: dtype string → Triton dtype constexpr
# ─────────────────────────────────────────────────────────────────
_DTYPE_MAP = {
    'float16':  tl.float16,
    'bfloat16': tl.bfloat16,
    'float32':  tl.float32,
}


# ─────────────────────────────────────────────────────────────────
#  Kernel 1: 3×3 depthwise conv + GELU (fused)
# ─────────────────────────────────────────────────────────────────
@triton.jit
def _dw_3x3_gelu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, C_in, H, W,
    BLOCK_C:  tl.constexpr,
    BLOCK_HW: tl.constexpr,
    DTYPE:    tl.constexpr,
):
    # pid_c  = output channel tile
    # pid_hw = spatial-position tile
    pid_c  = tl.program_id(0)
    pid_hw = tl.program_id(1)

    c_base  = pid_c  * BLOCK_C
    hw_base = pid_hw * BLOCK_HW

    c_offs  = c_base  + tl.arange(0, BLOCK_C)   # [BLOCK_C]
    hw_offs = hw_base + tl.arange(0, BLOCK_HW)  # [BLOCK_HW]
    c_mask  = c_offs  < C
    hw_mask = hw_offs < B * H * W

    HW = H * W
    # acc is [BLOCK_C, BLOCK_HW]
    acc = tl.zeros([BLOCK_C, BLOCK_HW], dtype=tl.float32)

    for kh in tl.static_range(3):
        for kw in tl.static_range(3):
            x_h = h_idx + (kh - 1)
            x_w = w_idx + (kw - 1)
            valid = (c_mask[:, None] &
                     hw_mask[None, :] &
                     (x_h >= 0)      & (x_h < H) &
                     (x_w >= 0)      & (x_w < W))
            # x[c, kernel_position]: load [BLOCK_C, BLOCK_HW]
            x_val = tl.load(x_ptr + x_h.to(tl.int32) * W + x_w.to(tl.int32),
                            mask=valid, other=0.0).to(tl.float32)
            # weight[c, kh, kw] — weight layout is [C_out, C_in, 3, 3]
            w_val = tl.load(w_ptr + c_offs.to(tl.float32) * C_in + kh * 3 + kw,
                             mask=c_mask, other=0.0).to(tl.float32)
            acc   = acc + x_val * w_val[:, None]

    # add bias
    b_val = tl.load(b_ptr + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    acc   = acc + b_val[:, None]

    # GELU:  x * 0.5 * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.7071067811865476
    gelu_f32 = acc * 0.5 * (1.0 + tl.math.erf(acc * inv_sqrt2))

    # Cast to the output dtype (float16 / bfloat16 / float32)
    gelu_out = gelu_f32 if DTYPE == tl.float32 else (
        gelu_f32.to(tl.float16) if DTYPE == tl.float16 else
        gelu_f32.to(tl.bfloat16)
    )

    # store [BLOCK_C, BLOCK_HW] in NCHW layout: out[c, h, w]
    out_idx = c_offs[:, None] * HW + h_idx[None, :] * W + w_idx[None, :]
    tl.store(out_ptr + out_idx, gelu_out,
             mask=c_mask[:, None] & hw_mask[None, :])


@triton.jit
def _dw_1x1_gelu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, C_in, C_out, H, W,
    BLOCK_BHW: tl.constexpr,
    BLOCK_K:   tl.constexpr,
    DTYPE:     tl.constexpr,
):
    """
    Tile over (spatial positions × C_out).
    M = B*H*W rows, each row dot-product [K=C_in] → [N=C_out].
    """
    pid_m  = tl.program_id(0)   # tile over M = B*H*W
    pid_n  = tl.program_id(1)   # output channel tile

    m_base = pid_m  * BLOCK_BHW
    k_base = pid_n  * BLOCK_K

    m_offs = m_base  + tl.arange(0, BLOCK_BHW)
    n_offs = k_base  + tl.arange(0, BLOCK_K)

    HW       = H * W
    m_mask   = m_offs < B * H * W
    n_mask   = n_offs < C_out

    # A[m, k] in NCHW:  x_ptr + (m//HW)*C_in*HW
    #                   + k*HW + (m%W)          + (m//W)*H*W
    # Simpler: pass strides
    x_idx = (m_offs // HW) * C_in * HW + n_offs * HW + (m_offs % HW)
    a = tl.load(x_ptr + x_idx, mask=m_mask[:, None] & n_mask[None, :], other=0.0)

    # B[k, n] = weight[n, k]  (NCHW weight: w_ptr[c_out*C_in + c_in])
    b_idx = n_offs[None, :] * C_in + m_offs[:, None]   # n_offs is C_out dim
    b     = tl.load(w_ptr + b_idx, mask=n_mask[None, :] & m_mask[:, None], other=0.0)

    # acc[BLOCK_BHW, BLOCK_K] = a @ b  (end up wanting [BLOCK_BHW, BLOCK_K] → [BLOCK_BHW, 1])
    # Instead, let me tile over n in the C_out direction:
    # We want each output element (m, n) = sum_k a[m,k] * b[k,n]
    # with a batch=1, so this is just broadcasting a[None, :] * b[:, None]

    # Reshape to [BLOCK_BHW, BLOCK_K] and [BLOCK_K, BLOCK_N] for dot:
    a_flat = tl.reshape(a, [BLOCK_BHW, BLOCK_K])
    b_flat = tl.reshape(b, [BLOCK_K, BLOCK_N])

    # Tensor-core-friendly product
    acc = tl.zeros([BLOCK_BHW, BLOCK_N], dtype=None)
    acc = tl.dot(a_flat, b_flat, acc)

    # bias
    bias_idx = n_offs   # [BLOCK_N]
    bias     = tl.load(b_ptr + bias_idx, mask=n_mask, other=0.0)
    acc      = acc + bias[None, :]

    # GELU
    inv_sqrt2 = 0.7071067811865476
    gelu_f32 = acc * 0.5 * (1.0 + tl.math.erf(acc * inv_sqrt2))

    # Cast to the output dtype (float16 / bfloat16 / float32)
    gelu_out = gelu_f32 if DTYPE == tl.float32 else (
        gelu_f32.to(tl.float16) if DTYPE == tl.float16 else
        gelu_f32.to(tl.bfloat16)
    )

    # out[B, C_out, H, W]:  same NCHW layout
    out_idx = (m_offs // HW)[:, None] * C_out * HW + n_offs[None, :] * HW + (m_offs % HW)
    tl.store(out_ptr + out_idx, gelu_out, mask=m_mask[:, None] & n_mask[None, :])


@torch.fx.wrap
def fused_dw_conv_gelu_dropout(bias, weight, x, route):
    """
    Common dispatch wrapper for all depthwise-conv + GELU variants.
    bias   : [C_out]
    weight : [C_out, C_in, kH, kW]
    x      : [B, C_in, H, W]
    route  : string describing which variant was matched
    """
    dtype = x.dtype
    dev   = x.device

    # ── route dispatch ──────────────────────────────────────────────
    if route == "3x3_128":
        C_out, C_in, kH, kW = weight.shape
        B, H, W = x.shape[0], x.shape[2], x.shape[3]
        _dw_3x3_gelu_kernel[
            lambda meta: (
                triton.cdiv(C_out,  1),
                triton.cdiv(B * H * W, 128),
            )
        ](
            x, weight, bias, out,
            B, C_in, H, W,
            BLOCK_C=1, BLOCK_HW=128,
            DTYPE=_DTYPE_MAP.get(str(x.dtype).split('.')[-1]),
        )
    elif route == "3x3_256":
        C_out, C_in, kH, kW = weight.shape
        B, H, W = x.shape[0], x.shape[2], x.shape[3]
        _dw_3x3_gelu_kernel[
            lambda meta: (
                triton.cdiv(C_out, 4),
                triton.cdiv(B * H * W, 64),
            )
        ](
            x, weight, bias, out,
            B, C_in, H, W,
            BLOCK_C=4, BLOCK_HW=64,
            DTYPE=_DTYPE_MAP.get(str(x.dtype).split('.')[-1]),
        )
    elif route == "3x3_512":
        C_out, C_in, kH, kW = weight.shape
        B, H, W = x.shape[0], x.shape[2], x.shape[3]
        _dw_3x3_gelu_kernel[
            lambda meta: (
                triton.cdiv(C_out, 8),
                triton.cdiv(B * H * W, 64),
            )
        ](
            x, weight, bias, out,
            B, C_in, H, W,
            BLOCK_C=8, BLOCK_HW=64,
            DTYPE=_DTYPE_MAP.get(str(x.dtype).split('.')[-1]),
        )
    elif route == "3x3_1024":
        C_out, C_in, kH, kW = weight.shape
        B, H, W = x.shape[0], x.shape[2], x.shape[3]
        _dw_3x3_gelu_kernel[
            lambda meta: (
                triton.cdiv(C_out, 32),
                triton.cdiv(B * H * W, 32),
            )
        ](
            x, weight, bias, out,
            B, C_in, H, W,
            BLOCK_C=32, BLOCK_HW=32,
            DTYPE=_DTYPE_MAP.get(str(x.dtype).split('.')[-1]),
        )
    elif route == "3x3_2048":
        C_out, C_in, kH, kW = weight.shape
        B, H, W = x.shape[0], x.shape[2], x.shape[3]
        _dw_3x3_gelu_kernel[
            lambda meta: (
                triton.cdiv(C_out, 64),
                triton.cdiv(B * H * W, 16),
            )
        ](
            x, weight, bias, out,
            B, C_in, H, W,
            BLOCK_C=64, BLOCK_HW=16,
            DTYPE=_DTYPE_MAP.get(str(x.dtype).split('.')[-1]),
        )
    elif route == "1x1_1":
        C_out, C_in, kH, kW = weight.shape
        B, H, W = x.shape[0], x.shape[2], x.shape[3]
        BLOCK_M, BLOCK_N, BLOCK_K = 256, 16, 16
        out = torch.empty(B, C_out, H, W, dtype=x.dtype, device=x.device)
        _dw_1x1_gelu_kernel[
            lambda meta: (
                triton.cdiv(B * H * W, BLOCK_M),
                triton.cdiv(C_out,  BLOCK_N),
            )
        ](
            x, weight, bias, out,
            B, C_in, C_out, HW,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            DTYPE=_DTYPE_MAP.get(str(x.dtype).split('.')[-1]),
        )
    else:
        # generic fallback
        raise ValueError(f"Unknown route: {route}")

    return x


# ─────────────────────────────────────────────────────────────────
#  Per-pass replacement functions (one per pattern)
# ─────────────────────────────────────────────────────────────────
def _replacement_func_128_p1_gelu():
    from .kernels import fused_dw_conv_gelu_dropout as _d
    return _d(bias, weight, x, "3x3_128")

def _replacement_func_128_p1_gelu_nd():
    from .kernels import fused_dw_conv_gelu_dropout as _d
    return _d(bias, weight, x, "3x3_128")

def _replacement_func_256_p1_gelu():
    from .kernels import fused_dw_conv_gelu_dropout as _d
    return _d(bias, weight, x, "3x3_256")

def _replacement_func_256_p1_gelu_nd():
    from .kernels import fused_dw_conv_gelu_dropout as _d
    return _d(bias, weight, x, "3x3_256")

def _replacement_func_512_p1_gelu():
    from .kernels import fused_dw_conv_gelu_dropout as _d
    return _d(bias, weight, x, "3x3_512")

def _replacement_func_512_p1_gelu_nd():
    from .kernels import fused_dw_conv_gelu_dropout as _d
    return _d(bias, weight, x, "3x3_512")

def _replacement_func_1024_p1_gelu():
    from .kernels import fused_dw_conv_gelu_dropout as _d
    return _d(bias, weight, x, "3x3_1024")

def _replacement_func_1024_p1_gelu_nd():
    from .kernels import fused_dw_conv_gelu_dropout as _d
    return _d(bias, weight, x, "3x3_1024")

def _replacement_func_2048_p1_gelu():
    from .kernels import fused_dw_conv_gelu_dropout as _d
    return _d(bias, weight, x, "3x3_2048")

def _replacement_func_2048_p1_gelu_nd():
    from .kernels import fused_dw_conv_gelu_dropout as _d
    return _d(bias, weight, x, "3x3_2048")

def _replacement_func_fastvit_p0_no_gelu():
    from .kernels import fused_dw_conv_gelu_dropout as _d
    return _d(bias, weight, x, "1x1_1")