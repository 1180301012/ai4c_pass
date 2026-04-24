import torch
import triton
import triton.language as tl


@triton.jit
def fused_scale_add_bn_kernel(
    conv_out_ptr,  # [B, C, H, W] contiguous
    gamma_ptr,     # [C, 1, 1]   – layer-scale weight (NCHW contiguous: index c)
    residual_ptr,  # [B, C, H, W] contiguous
    mean_ptr,      # [C]
    var_ptr,       # [C]
    bn_weight_ptr, # [C]
    bn_bias_ptr,   # [C]
    out_bn_ptr,    # [B, C, H, W] contiguous  (batch_norm output)
    out_sum_ptr,   # [B, C, H, W] contiguous  (residual + scaled conv output)
    NCHW,          # total elements = N*C*H*W
    HW,            # H*W
    C,             # number of channels
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < NCHW

    # channel index: layout is [N, C, H, W] so
    #   channel = (flat_idx // HW) % C
    c_idx = (offsets // HW) % C

    # per-channel params (broadcast within block)
    means = tl.load(mean_ptr     + c_idx, mask=mask, other=0.0).to(tl.float32)
    vars  = tl.load(var_ptr      + c_idx, mask=mask, other=1.0).to(tl.float32)
    bn_ws = tl.load(bn_weight_ptr + c_idx, mask=mask, other=1.0).to(tl.float32)
    bn_bs = tl.load(bn_bias_ptr   + c_idx, mask=mask, other=0.0).to(tl.float32)
    # gamma has same dtype as activations
    gammas_native  = tl.load(gamma_ptr + c_idx, mask=mask, other=0.0)
    gammas_f32     = tl.load(gamma_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)

    inv_stds = 1.0 / tl.sqrt(vars + eps)

    # loads – keep native dtype
    xs = tl.load(conv_out_ptr  + offsets, mask=mask, other=0.0)
    rs = tl.load(residual_ptr  + offsets, mask=mask, other=0.0)

    # compute in fp32 for batch_norm accuracy
    x_f32  = xs.to(tl.float32)
    r_f32  = rs.to(tl.float32)

    # layer-scale multiply  (dropout(p=0,training=False) is identity)
    sum_val  = r_f32 + x_f32 * gammas_f32

    # batch_norm inference
    bn_val = (sum_val - means) * inv_stds * bn_ws + bn_bs

    tl.store(out_sum_ptr + offsets, sum_val.to(xs.dtype), mask=mask)
    tl.store(out_bn_ptr  + offsets, bn_val.to(xs.dtype),  mask=mask)


@torch.fx.wrap
def fused_scale_add_bn(conv_out, gamma, residual, running_mean, running_var,
                       bn_weight, bn_bias):
    N, C, H, W = conv_out.shape
    NCHW = N * C * H * W
    HW   = H * W

    out_sum = torch.empty_like(conv_out)
    out_bn  = torch.empty_like(conv_out)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(NCHW, BLOCK_SIZE),)

    fused_scale_add_bn_kernel[grid](
        conv_out, gamma, residual,
        running_mean, running_var, bn_weight, bn_bias,
        out_bn, out_sum,
        NCHW, HW, C,
        1e-5,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out_bn, out_sum


# ── pattern: matches scale(mul) + residual-add + batch_norm ───────────────────
# dropout(p=0, training=False) is identity; it may be optimized away before pass runs
def pattern(conv_out, gamma, residual, running_mean, running_var, bn_weight, bn_bias):
    tmp_9  = conv_out * gamma
    tmp_10 = residual + tmp_9
    tmp_11 = torch.nn.functional.batch_norm(
        tmp_10, running_mean, running_var, bn_weight, bn_bias,
        False, 0.1, 1e-05
    )
    return tmp_11, tmp_10


def replacement_args(conv_out, gamma, residual,
                     running_mean, running_var, bn_weight, bn_bias):
    return (conv_out, gamma, residual, running_mean, running_var, bn_weight, bn_bias)


def replacement_func():
    return fused_scale_add_bn