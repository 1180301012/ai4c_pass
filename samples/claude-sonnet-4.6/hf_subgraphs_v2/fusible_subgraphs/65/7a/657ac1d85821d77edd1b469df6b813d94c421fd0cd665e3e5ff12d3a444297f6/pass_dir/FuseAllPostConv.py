import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: dropout(p=0) + scale-mul + residual-add + inference batch-norm
#   Matches the FULL post-conv sequence and returns BOTH observable outputs.
#   The eps constant is passed as a float32 tensor pointer (not a Python
#   float literal) to guarantee fp32 arithmetic in tl.rsqrt – the literal
#   1e-5 is a Python float64 which breaks tl.rsqrt on older Triton builds.
# ---------------------------------------------------------------------------
def pattern(conv_out, gamma, residual,
            running_mean, running_var, bn_weight, bn_bias):
    dropped = torch.nn.functional.dropout(conv_out, 0.0, False, False)
    scaled  = dropped * gamma
    pre_bn  = residual + scaled
    post_bn = torch.nn.functional.batch_norm(
        pre_bn, running_mean, running_var, bn_weight, bn_bias,
        False, 0.1, 1e-05)
    return (post_bn, pre_bn)


def replacement_args(conv_out, gamma, residual,
                     running_mean, running_var, bn_weight, bn_bias):
    return (conv_out, gamma, residual,
            running_mean, running_var, bn_weight, bn_bias)


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------
@triton.jit
def _fused_all_kernel(
    x_ptr,        # conv2d output (raw, before dropout)
    gamma_ptr,    # layer-scale  [C, 1, 1]
    res_ptr,      # residual     [B, C, H, W]
    mean_ptr,     # BN running_mean [C]
    var_ptr,      # BN running_var  [C]
    w_ptr,        # BN weight       [C]
    b_ptr,        # BN bias         [C]
    eps_ptr,      # float32 scalar tensor (size 1) – avoids fp64 literal
    out_pre_ptr,  # pre_bn  output
    out_post_ptr, # post_bn output
    HW,
    C,
    BLOCK_HW: tl.constexpr,
):
    bc_pid  = tl.program_id(0)   # = b * C + c
    hw_tile = tl.program_id(1)

    # Channel index: one integer mod per BLOCK (not per element)
    c    = bc_pid % C
    base = bc_pid * HW

    hw_offs = hw_tile * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask    = hw_offs < HW
    offs    = base + hw_offs

    # --- Load eps as float32 from tensor (no Python float literal) ---
    eps_f32 = tl.load(eps_ptr).to(tl.float32)

    # --- Per-channel scalars (one load each per block) ---
    gamma_val = tl.load(gamma_ptr + c)
    mean_f32  = tl.load(mean_ptr  + c).to(tl.float32)
    var_f32   = tl.load(var_ptr   + c).to(tl.float32)
    w_f32     = tl.load(w_ptr     + c).to(tl.float32)
    b_f32     = tl.load(b_ptr     + c).to(tl.float32)

    # --- Pre-compute BN affine params in fp32 (once per block) ---
    inv_std   = tl.rsqrt(var_f32 + eps_f32)   # both fp32 ✓
    bn_scale  = w_f32 * inv_std
    bn_offset = b_f32 - mean_f32 * bn_scale

    # --- Vector loads (coalesced) ---
    x = tl.load(x_ptr   + offs, mask=mask, other=0.)
    r = tl.load(res_ptr + offs, mask=mask, other=0.)

    # dropout(p=0, training=False) is identity → skip
    # pre_bn = residual + conv_out * gamma
    pre_bn = r + x * gamma_val

    # post_bn = BN(pre_bn) in fp32
    post_f32 = pre_bn.to(tl.float32) * bn_scale + bn_offset
    post_bn  = post_f32.to(x.dtype)

    tl.store(out_pre_ptr  + offs, pre_bn,  mask=mask)
    tl.store(out_post_ptr + offs, post_bn, mask=mask)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_dropout_scale_add_bn(conv_out, gamma, residual,
                                running_mean, running_var,
                                bn_weight, bn_bias):
    B, C, H, W = conv_out.shape
    HW       = H * W
    BC       = B * C
    BLOCK_HW = 1024

    # Pass eps as a float32 tensor – guarantees fp32 inside the kernel
    eps_t = torch.full((1,), 1e-5, dtype=torch.float32, device=conv_out.device)

    out_pre  = torch.empty_like(conv_out)
    out_post = torch.empty_like(conv_out)

    hw_tiles = (HW + BLOCK_HW - 1) // BLOCK_HW
    grid     = (BC, hw_tiles)

    _fused_all_kernel[grid](
        conv_out,
        gamma,           # [C,1,1] – C contiguous floats
        residual,
        running_mean,
        running_var,
        bn_weight,
        bn_bias,
        eps_t,
        out_pre,
        out_post,
        HW, C,
        BLOCK_HW=BLOCK_HW,
    )

    return (out_post, out_pre)   # (tmp_11, tmp_10)


def replacement_func():
    return fused_dropout_scale_add_bn