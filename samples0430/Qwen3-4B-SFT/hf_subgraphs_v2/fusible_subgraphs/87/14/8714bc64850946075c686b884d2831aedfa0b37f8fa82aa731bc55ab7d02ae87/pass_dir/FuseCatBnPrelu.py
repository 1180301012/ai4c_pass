import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 256},  num_warps=4),
        triton.Config({'BLOCK_HW': 512},  num_warps=4),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8),
        triton.Config({'BLOCK_HW': 2048}, num_warps=8),
        triton.Config({'BLOCK_HW': 4096}, num_warps=8),
    ],
    key=['C_src', 'HW'],
)
@triton.jit
def fused_cat_bn_prelu_kernel(
    in6_ptr, conv_ptr, w_ptr, rm_ptr, rv_ptr, bnw_ptr, bnb_ptr,
    out_ptr,
    B, C_src, C_conv, C_total, HW,
    eps,
    BLOCK_HW: tl.constexpr,
):
    """
    Fused: cat([in_6, conv], dim=1) + BatchNorm(inference) + PReLU

    Grid: (B * C_total,  ceil(HW / BLOCK_HW))

    - programs with pid_ac < C_total  → output channels from in_6
      src_base = pid_b * C_src * HW   (reads from in_6's C_src channels)
    - programs with pid_ac >= C_total → output channels from conv
      src_base = pid_b * C_conv * HW  (reads from conv's C_conv channels)
      pid_local = pid_ac - C_total

    By mapping both halves to the same pid_ac range [0, C_total), arithmetic
    on C_src / C_conv is never needed, and the only two-pointer load is
    resolved at the PROGRAM level via integer division — no dynamic if / tl.where
    on runtime scalars.
    """
    pid_ac  = tl.program_id(0)   # position within [0, C_total) combined with batch
    pid_hw  = tl.program_id(1)   # spatial-block index

    # Split pid_ac into batch index and channel within this source tensor
    HW_per_batch = C_src * HW   # elements in one batch of this source tensor
    batch_idx  = pid_ac // C_total
    src_ch     = pid_ac % C_total          # in [0, C_total)

    hw_start   = pid_hw * BLOCK_HW
    hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
    hw_mask    = hw_offsets < HW

    # ── Per-channel affine parameters (fp32 for numerical stability) ─────────
    rm     = tl.load(rm_ptr   + src_ch).to(tl.float32)
    rv_val = tl.load(rv_ptr   + src_ch).to(tl.float32)
    bnw    = tl.load(bnw_ptr  + src_ch).to(tl.float32)
    bnb    = tl.load(bnb_ptr  + src_ch).to(tl.float32)
    pw     = tl.load(w_ptr    + src_ch).to(tl.float32)

    # ── Source tensor pointer ────────────────────────────────────────────────
    # c_local is always in [0, C_src-1] for in6 AND [0, C_conv-1] for conv:
    c_local = src_ch % C_src    # pid_ac % C_src works for both halves
    is_in6  = pid_ac < C_total
    src_base = batch_idx * (C_src * HW if is_in6 else C_conv * HW)
    src_off  = src_base + c_local * HW + hw_offsets

    # ── Load from the correct source (only one is unmasked) ─────────────────
    x_in6  = tl.load(in6_ptr  + src_off,  mask=hw_mask & is_in6,  other=0.0)
    x_conv = tl.load(conv_ptr + src_off,   mask=hw_mask & ~is_in6, other=0.0)
    x = x_in6 + x_conv   # exactly one term is non-zero

    # ── Batch-norm (inference, fp32) ────────────────────────────────────────
    inv_std = bnw / tl.sqrt(rv_val + eps)
    y = x * inv_std + (bnb - rm * inv_std)

    # ── PReLU ────────────────────────────────────────────────────────────────
    y_prelu = tl.where(y >= 0.0, y, y * pw)

    # ── Write into output [B, C_total, H, W] ────────────────────────────────
    out_ch   = src_ch + C_src * (batch_idx > 0)   # channel in the cat output
    out_off  = batch_idx * C_total * HW + out_ch * HW + hw_offsets
    tl.store(out_ptr + out_off, y_prelu, mask=hw_mask)


# ── Pattern ──────────────────────────────────────────────────────────────────

def pattern(in_6, conv, running_mean, running_var, bn_weight, bn_bias, prelu_weight):
    """
    Match: cat([in_6, conv], dim=1) → batch_norm(inference) → prelu
    in_6            : [B, C_src, H, W]       (loc tensor)
    conv            : [B, C_src, H, W]       (conv2d output)
    running_mean    : [C_total]
    running_var     : [C_total]
    bn_weight (γ)   : [C_total]
    bn_bias   (β)   : [C_total]
    prelu_weight    : [C_total]
    """
    tmp_7  = torch.cat([in_6, conv], 1)
    tmp_8  = torch.nn.functional.batch_norm(
                 tmp_7, running_mean, running_var, bn_weight, bn_bias,
                 False, 0.1, 0.001)
    tmp_9  = torch.prelu(tmp_8, prelu_weight)
    return tmp_9


def replacement_args(in_6, conv, running_mean, running_var, bn_weight, bn_bias, prelu_weight):
    return (in_6, conv, running_mean, running_var, bn_weight, bn_bias, prelu_weight)


# ── Optimised replacement ─────────────────────────────────────────────────────

@torch.fx.wrap
def fused_cat_bn_prelu(in_6, conv, running_mean, running_var, bn_weight, bn_bias, prelu_weight):
    B  = in_6.shape[0]
    C_src  = in_6.shape[1]     # 64
    C_conv = conv.shape[1]     # 64
    C_total = C_src + C_conv   # 128
    H  = in_6.shape[2]
    W  = in_6.shape[3]
    HW = H * W

    # Output shape: [B, C_total, H, W]  (not like in_6 because we add C_conv channels)
    out = torch.empty((B, C_total, H, W), dtype=in_6.dtype, device=in_6.device)

    def grid(META):
        return (B * C_total, triton.cdiv(HW, META['BLOCK_HW']))

    fused_cat_bn_prelu_kernel[grid](
        in_6, conv,
        prelu_weight, running_mean, running_var, bn_weight, bn_bias,
        out,
        B, C_src, C_conv, C_total, HW,
        1e-3,           # eps = 0.001
    )

    return out


def replacement_func():
    return fused_cat_bn_prelu