import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel: fuse dropout(p=0.0) + layer-scale * + residual + + BN-infer
# Compute in fp32 for precision; stores are cast back to the pointer dtype.
# ---------------------------------------------------------------------------

@triton.jit
def _dropout_scale_add_bn_kernel(
    conv_out_ptr,   # [B, C, H, W] – conv2d output (dropout is identity at p=0)
    gamma_ptr,      # [C, 1, 1]   – layer-scale γ  (stored as C contiguous floats)
    residual_ptr,   # [B, C, H, W] – skip-connection input
    bn_mean_ptr,    # [C]          – BN running mean
    bn_var_ptr,     # [C]          – BN running var
    bn_weight_ptr,  # [C]          – BN affine weight
    bn_bias_ptr,    # [C]          – BN affine bias
    tmp10_ptr,      # [B, C, H, W] – output: residual + scaled_conv  (also returned)
    tmp11_ptr,      # [B, C, H, W] – output: batch_norm(tmp10)       (also returned)
    N,              # total elements = B * C * H * W
    HW,             # H * W  (spatial size per channel)
    C,              # number of channels
    BLOCK_SIZE: tl.constexpr,
):
    pid     = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask    = offsets < N

    # channel index for NCHW layout: flat_idx -> c = (flat // HW) % C
    c_idx = (offsets // HW) % C

    # ---- load input tensors in native dtype --------------------------------
    conv_out_raw = tl.load(conv_out_ptr  + offsets, mask=mask, other=0.0)
    residual_raw = tl.load(residual_ptr  + offsets, mask=mask, other=0.0)
    gamma_raw    = tl.load(gamma_ptr     + c_idx,   mask=mask, other=1.0)
    bn_mean_raw  = tl.load(bn_mean_ptr   + c_idx,   mask=mask, other=0.0)
    bn_var_raw   = tl.load(bn_var_ptr    + c_idx,   mask=mask, other=1.0)
    bn_w_raw     = tl.load(bn_weight_ptr + c_idx,   mask=mask, other=1.0)
    bn_b_raw     = tl.load(bn_bias_ptr   + c_idx,   mask=mask, other=0.0)

    # ---- upcast to fp32 for numerical precision ---------------------------
    co  = conv_out_raw.to(tl.float32)
    res = residual_raw.to(tl.float32)
    g   = gamma_raw.to(tl.float32)
    mu  = bn_mean_raw.to(tl.float32)
    var = bn_var_raw.to(tl.float32)
    bw  = bn_w_raw.to(tl.float32)
    bb  = bn_b_raw.to(tl.float32)

    # ---- fused computation ------------------------------------------------
    # dropout(p=0, training=False) is identity → skip

    # tmp9  = conv_out * gamma   (layer scale)
    scaled = co * g

    # tmp10 = residual + scaled  (residual add)
    tmp10_f32 = res + scaled

    # tmp11 = batch_norm_inference(tmp10)
    #       = (tmp10 - mu) / sqrt(var + ε) * bw + bb
    inv_std    = tl.rsqrt(var + 1e-5)
    tmp11_f32  = (tmp10_f32 - mu) * inv_std * bw + bb

    # ---- cast back to original dtype and store ----------------------------
    out_dtype = conv_out_raw.dtype
    tl.store(tmp10_ptr + offsets, tmp10_f32.to(out_dtype), mask=mask)
    tl.store(tmp11_ptr + offsets, tmp11_f32.to(out_dtype), mask=mask)


# ---------------------------------------------------------------------------
# Python wrapper (must be @torch.fx.wrap so FX graph tracing skips into it)
# ---------------------------------------------------------------------------

@torch.fx.wrap
def _fused_dropout_scale_add_bn(
    conv_out,   # [B, C, H, W]
    gamma,      # [C, 1, 1]
    residual,   # [B, C, H, W]
    bn_mean,    # [C]
    bn_var,     # [C]
    bn_weight,  # [C]
    bn_bias,    # [C]
):
    B, C, H, W = conv_out.shape
    N  = B * C * H * W
    HW = H * W

    tmp10 = torch.empty_like(conv_out)
    tmp11 = torch.empty_like(conv_out)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)

    _dropout_scale_add_bn_kernel[grid](
        conv_out, gamma, residual,
        bn_mean, bn_var, bn_weight, bn_bias,
        tmp10, tmp11,
        N, HW, C,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # return order must match model: (tmp_11, tmp_10)
    return (tmp11, tmp10)


# ---------------------------------------------------------------------------
# Pattern / replacement interface required by the AI4C pass framework
# ---------------------------------------------------------------------------

def pattern(conv_out, gamma, residual, bn_running_mean, bn_running_var, bn_weight, bn_bias):
    """
    Matches:
        tmp_8  = dropout(conv_out, 0.0, False, False)   # p=0, training=False → identity
        tmp_9  = tmp_8 * gamma
        tmp_10 = residual + tmp_9
        tmp_11 = batch_norm(tmp_10, ..., False, 0.1, 1e-05)
    Returns both observable outputs: (tmp_11, tmp_10)
    """
    tmp_8  = torch.nn.functional.dropout(conv_out, 0.0, False, False)
    tmp_9  = tmp_8 * gamma
    tmp_10 = residual + tmp_9
    tmp_11 = torch.nn.functional.batch_norm(
        tmp_10, bn_running_mean, bn_running_var, bn_weight, bn_bias,
        False, 0.1, 1e-05,
    )
    return (tmp_11, tmp_10)


def replacement_args(conv_out, gamma, residual, bn_running_mean, bn_running_var, bn_weight, bn_bias):
    return (conv_out, gamma, residual, bn_running_mean, bn_running_var, bn_weight, bn_bias)


def replacement_func():
    return _fused_dropout_scale_add_bn