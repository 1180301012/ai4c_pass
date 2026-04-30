"""
Shared Triton kernel for fusing:
  1x1 grouped conv (scale + bias) + 2x add + batch_norm (inference) + spatial mean
into a single memory pass.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _fused_conv_bn_mean_kernel(
    x0_ptr, x1_ptr,                  # [B, C, H, W] inputs
    w_conv_ptr, b_conv_ptr,          # [C] per-channel conv weight & bias
    running_mean_ptr, running_var_ptr,# [C] BN running stats
    gamma_ptr, beta_ptr,             # [C] BN weight & bias
    out_ptr,                         # [B, C, H, W] output
    mean_out_ptr,                    # [B, C] mean output (caller squeezes to [B,C,1,1])
    C, HW,
    eps,
    BLOCK_HW: tl.constexpr,
):
    """
    One program per (b, c) pair.
    Processes all HW spatial elements in chunks of BLOCK_HW.
    Computes: out = alpha * x0 + beta_const + x1
              mean = sum(out) / HW
    where:
      alpha = gamma / sqrt(rv + eps)
      beta_const = beta - alpha * rm
    """
    bc = tl.program_id(0)
    c = bc % C

    # --- load per-channel params and precompute linear transform ---
    w   = tl.load(w_conv_ptr  + c).to(tl.float32)
    bc_ = tl.load(b_conv_ptr  + c).to(tl.float32)
    rm  = tl.load(running_mean_ptr + c).to(tl.float32)
    rv  = tl.load(running_var_ptr  + c).to(tl.float32)
    gm  = tl.load(gamma_ptr     + c).to(tl.float32)
    bt  = tl.load(beta_ptr      + c).to(tl.float32)

    inv_std = 1.0 / tl.sqrt(rv + eps)
    alpha   = gm * inv_std          # combined BN scale
    beta_c  = bt - alpha * rm       # combined BN bias

    base = bc * HW
    acc  = tl.zeros([BLOCK_HW], dtype=tl.float32)

    # --- loop over spatial tiles ---
    for start in range(0, HW, BLOCK_HW):
        offsets = start + tl.arange(0, BLOCK_HW)
        mask    = offsets < HW

        x0 = tl.load(x0_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(x1_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)

        # conv(C) + add + add + batch_norm(inference)
        tmp = x0 * w + bc_ + x1
        result = tmp * alpha + beta_c

        tl.store(out_ptr + base + offsets, result, mask=mask)

        acc += result * mask.to(tl.float32)

    mean_val = tl.sum(acc) / HW
    tl.store(mean_out_ptr + bc, mean_val)


@torch.fx.wrap
def fused_conv_bn_mean(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, groups):
    """
    in_0 : running_mean [C]
    in_1 : running_var  [C]
    in_2 : bn_bias      [C]
    in_3 : bn_weight    [C]
    in_4 : conv_bias    [C]
    in_5 : conv_weight  [C, 1, 1, 1]
    in_6 : first input  [B, C, H, W]
    in_7 : second input [B, C, H, W]
    groups : number of groups (== C for 1x1 conv)
    returns (out [B,C,H,W], mean_out [B,C,1,1])
    """
    B, C, H, W = in_6.shape
    HW = H * W
    device = in_6.device
    dtype  = in_6.dtype

    out      = torch.empty_like(in_6)
    mean_out = torch.empty((B * C,), dtype=dtype, device=device)

    BLOCK_HW = 512

    _fused_conv_bn_mean_kernel[(B * C,)](
        in_6, in_7,
        in_5, in_4,             # conv_weight [C,1,1,1], conv_bias [C]
        in_0, in_1,             # running_mean, running_var
        in_3, in_2,             # bn_weight (gamma), bn_bias (beta)
        out, mean_out,
        C, HW,
        1e-5,
        BLOCK_HW=BLOCK_HW,
    )

    return out, mean_out.reshape(B, C, 1, 1)