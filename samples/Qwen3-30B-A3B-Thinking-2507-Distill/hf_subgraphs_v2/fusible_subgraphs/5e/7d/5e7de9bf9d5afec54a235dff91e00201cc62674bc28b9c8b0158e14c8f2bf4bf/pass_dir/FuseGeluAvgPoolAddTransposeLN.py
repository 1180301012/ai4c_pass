import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: gelu + avg_pool1d + slice + slice + add + transpose + layer_norm
#          + dropout(training=False)  — all fused into one Triton kernel
#
# conv1d_out : [B, C, T]   – output of torch.conv1d (in_4 weight, stride=2)
# in_3       : [B, C, T_in] – original input tensor (fed into avg_pool1d)
# in_1       : [C]          – layer-norm weight
# in_0       : [C]          – layer-norm bias
#
# Key note: T_conv ≠ T_in//2.  With padding=15, kernel=31, stride=2:
#   T_conv = floor((T_in + 2*15 - 31) / 2) + 1  = 125 for T_in=249
# ---------------------------------------------------------------------------

def pattern(conv1d_out, in_3, in_1, in_0):
    tmp_4 = torch.nn.functional.gelu(conv1d_out)
    tmp_5 = torch.avg_pool1d(in_3, (2,), (2,), (0,), False, True)
    tmp_6 = tmp_5[(Ellipsis, slice(None, 124, None))]
    tmp_7 = tmp_4[(Ellipsis, slice(None, 124, None))]
    tmp_8 = tmp_6 + tmp_7
    tmp_9 = tmp_8.transpose(1, 2)
    tmp_10 = torch.nn.functional.layer_norm(tmp_9, (768,), in_1, in_0, 1e-05)
    tmp_11 = torch.nn.functional.dropout(tmp_10, 0.1, False, False)
    return tmp_11


def replacement_args(conv1d_out, in_3, in_1, in_0):
    return (conv1d_out, in_3, in_1, in_0)


# ---------------------------------------------------------------------------
# Triton kernel
#
# Grid: (B * T,)  — one program per (batch, output-time-step) pair
# Each program:
#   1. Loads conv1d_out[b, 0:C, t]  (C elements, stride_T_conv)
#   2. Computes GELU
#   3. Loads in_3[b, 0:C, 2t] and in_3[b, 0:C, 2t+1]  (C each)
#   4. Averages the two in_3 loads
#   5. Adds GELU result + avg pool result
#   6. Applies LayerNorm over the C channels
#   7. Stores to out[b, t, 0:C]  (contiguous, C elements)
#
# BLOCK_C must be >= C and a power of two (1024 >= 768).
# Masked positions (C .. BLOCK_C-1) are zeroed so they don't corrupt the
# LayerNorm mean/variance computation.
# ---------------------------------------------------------------------------

@triton.jit
def _fused_kernel(
    conv1d_ptr,     # [B, C, T_conv]
    in3_ptr,        # [B, C, T_in]
    weight_ptr,     # [C]
    bias_ptr,       # [C]
    out_ptr,        # [B, T_conv, C]
    T_in,           # length of in_3 along dim-2  (249)
    T_conv,         # output length from conv1d   (124 or 125)
    C,              # 768
    conv_stride_T,  # stride of conv1d output along dim-2
    in3_stride_T,   # stride of in_3 along dim-2
    out_stride_T,   # stride of out along dim-2  (== C for contiguous)
    BLOCK_C: tl.constexpr,
):
    pid  = tl.program_id(0)
    b    = pid // T_conv
    t    = pid % T_conv

    c_idx = tl.arange(0, BLOCK_C)
    mask  = c_idx < C

    # ---- Load conv1d output and apply GELU (float32 accumulation) ----
    conv_off = b * C * conv_stride_T + c_idx * conv_stride_T + t
    x = tl.load(conv1d_ptr + conv_off, mask=mask, other=0.0).to(tl.float32)

    # GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    x_gelu = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))

    # ---- Average pool: (in_3[b,c,2t] + in_3[b,c,2t+1]) / 2 ----
    in3_base = b * C * in3_stride_T + c_idx * in3_stride_T
    x_avg = (tl.load(in3_ptr + in3_base + 2 * t,   mask=mask, other=0.0).to(tl.float32) +
             tl.load(in3_ptr + in3_base + 2 * t + 1, mask=mask, other=0.0).to(tl.float32)) * 0.5

    # ---- Add gelu + avg ----
    x = x_gelu + x_avg

    # ---- LayerNorm over C channels ----
    # Mean: masked (padded) positions are 0.0, so sum/C is exact.
    mean = tl.sum(x, axis=0) / C

    # Zero out masked positions before computing variance so padding slots
    # do NOT contribute (-mean)^2 to the variance sum.
    x_centered = tl.where(mask, x - mean, 0.0)
    var   = tl.sum(x_centered * x_centered, axis=0) / C
    rstd  = 1.0 / tl.sqrt(var + 1e-5)

    # Normalized: padding positions will be discarded at the store step.
    x_norm = x_centered * rstd

    # Affine transform
    w  = tl.load(weight_ptr + c_idx, mask=mask, other=1.0).to(tl.float32)
    b_ = tl.load(bias_ptr  + c_idx, mask=mask, other=0.0).to(tl.float32)
    out = x_norm * w + b_

    # ---- Store transposed: out[b, t, c] ----
    out_off = b * out_stride_T * C + t * out_stride_T + c_idx
    tl.store(out_ptr + out_off, out.to(conv1d_ptr.dtype.element_ty), mask=mask)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_gelu_avgpool_add_ln(conv1d_out, in_3, weight, bias):
    """
    conv1d_out : [B, C, T_conv]   – output of conv1d (stride=2, pad=15, k=31)
    in_3       : [B, C, T_in]     – original input fed into avg_pool1d
    weight     : [C]   (in_1)
    bias       : [C]   (in_0)
    returns    : [B, T_conv, C]

    T_conv is computed with the PyTorch conv formula:
        T_conv = floor((T_in + 2*pad - k) / stride) + 1
    not the incorrect floor((T_in - k) / stride) + 1 that yields 124.
    For T_in=249, pad=15, k=31, stride=2:  (249+30-31)//2+1 = 125 ✓
    """
    # conv1d_out shape: [B, C, 125] at runtime → strides (C*125, 125, 1)
    # in_3 shape      : [B, C, 249]       → strides (C*249, 249, 1)
    # Output must be   : [B, 124, C]       (matches FX-graph recorded shape)
    B    = conv1d_out.shape[0]
    C    = conv1d_out.shape[1]
    T_in = in_3.shape[2]
    T_conv = 124   # FX-graph metadata (output dim-1 shape expected by framework)

    out = torch.empty(B, T_conv, C, dtype=conv1d_out.dtype, device=conv1d_out.device)

    # Use actual tensor strides — they differ from T_conv/T_in values:
    #   conv1d_out.stride(2) = 125  (not T_conv=124)
    #   in_3.stride(2)       = 249  (not T_in=249, but that *is* 249 here)
    #   out.stride(1)        = C    (contiguous [B, T_conv, C])
    conv_stride_T = conv1d_out.stride(2)   # actual stride along dim-2 of conv output
    in3_stride_T  = in_3.stride(2)         # actual stride along dim-2 of in_3
    out_stride_T  = out.stride(1)          # = C

    _fused_kernel[(B * T_conv,)](
        conv1d_out, in_3, weight, bias, out,
        T_in, T_conv, C,
        conv1d_out.stride(2),   # conv_stride_T
        in_3.stride(2),         # in3_stride_T
        out_stride_T,           # out_stride_T = C  (contiguous [B,T,C])
        BLOCK_C=1024,           # must be provided explicitly without autotuner
        num_warps=8,
    )
    return out


def replacement_func():
    return fused_gelu_avgpool_add_ln