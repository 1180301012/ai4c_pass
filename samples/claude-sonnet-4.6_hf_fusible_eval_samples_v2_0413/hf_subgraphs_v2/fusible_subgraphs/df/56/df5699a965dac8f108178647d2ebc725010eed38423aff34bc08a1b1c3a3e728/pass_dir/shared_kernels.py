"""
Shared Triton kernels for LiteHRNet channel-shuffle optimization.

Two single-output pass types, ALL sharing ONE replacement function:
  FuseChannelShuffle1_B{N}: cat(in2,in4)+shuffle → [B,40,64,48]
  FuseChannelShuffle2_B{N}: cat(in3,x)+shuffle   → [B,80,32,24]
Both use fused_shuffle_impl (generic, works for any [B,C,H,W] inputs).
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel: full channel interleave
#   a_ptr, b_ptr : [B, C_IN, HW] each
#   out_ptr      : [B, 2*C_IN, HW]
#     out[b, 2k,   hw] = a[b, k, hw]
#     out[b, 2k+1, hw] = b[b, k, hw]
#   Grid: (B * C_IN, ceil(HW / BLOCK))
# ---------------------------------------------------------------------------
@triton.jit
def interleave_full_kernel(
    a_ptr, b_ptr, out_ptr,
    B, C_IN, HW,
    BLOCK: tl.constexpr,
):
    pid_bc = tl.program_id(0)
    pid_hw = tl.program_id(1)

    b  = pid_bc // C_IN
    k  = pid_bc  % C_IN

    hw0     = pid_hw * BLOCK
    hw_offs = hw0 + tl.arange(0, BLOCK)
    mask    = hw_offs < HW

    src_off = b * (C_IN * HW) + k * HW + hw_offs
    a_vals  = tl.load(a_ptr + src_off, mask=mask, other=0.0)
    b_vals  = tl.load(b_ptr + src_off, mask=mask, other=0.0)

    out_base     = b * (2 * C_IN * HW)
    out_even_off = out_base + (2 * k)     * HW + hw_offs
    out_odd_off  = out_base + (2 * k + 1) * HW + hw_offs

    tl.store(out_ptr + out_even_off, a_vals, mask=mask)
    tl.store(out_ptr + out_odd_off,  b_vals, mask=mask)


# ---------------------------------------------------------------------------
# Single generic replacement wrapper used by ALL shuffle passes
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_shuffle_impl(a, b):
    """
    Generic channel interleave: cat([a, b], dim=1) + shuffle.
    a, b : [B, C, H, W]  (same shape)
    out  : [B, 2C, H, W]
    Works for both stream-1 (C=20,H=64,W=48) and stream-2 (C=40,H=32,W=24).
    """
    B   = a.shape[0]
    C   = a.shape[1]
    H   = a.shape[2]
    W   = a.shape[3]
    HW  = H * W
    out = torch.empty((B, C * 2, H, W), dtype=a.dtype, device=a.device)
    BLK  = 1024
    n_hw = (HW + BLK - 1) // BLK
    interleave_full_kernel[(B * C, n_hw)](a, b, out, B, C, HW, BLOCK=BLK)
    return out


# Aliases so pass files that import the old names still work
fused_shuffle1_impl = fused_shuffle_impl
fused_shuffle2_impl = fused_shuffle_impl



# ---------------------------------------------------------------------------
# Kernel: sigmoid attention from 1-by-1 conv (used by FuseConvSigmoidMul)
#   conv_ptr : [B, C, 1, 1]  (conv2d output, accessed as [B, C])
#   x_ptr    : [B, C, H, W]  (in_5, to be multiplied by sigmoid)
#   out_ptr  : [B, C, H, W]
#   N = B*C*H*W
# ---------------------------------------------------------------------------
@triton.jit
def fused_sigmoid_mul_kernel(
    conv_ptr, x_ptr, out_ptr,
    N, C, HW,
    BLOCK: tl.constexpr,
):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    # decode b, c from flat index
    b_c_hw = offs              # b*(C*HW) + c*HW + hw
    c_hw   = b_c_hw % (C * HW)
    c      = c_hw // HW
    b      = b_c_hw // (C * HW)

    # conv_out[b, c, 0, 0]  =>  flat index b*C + c  (squeezed [B,C])
    conv_val = tl.load(conv_ptr + b * C + c, mask=mask, other=0.0).to(tl.float32)
    sig      = 1.0 / (1.0 + tl.exp(-conv_val))

    x_val  = tl.load(x_ptr + offs, mask=mask, other=0.0)
    result = (x_val.to(tl.float32) * sig).to(x_val.dtype)
    tl.store(out_ptr + offs, result, mask=mask)


# ---------------------------------------------------------------------------
# Kernel: full channel interleave
#   a_ptr  : [B, C_IN, HW]
#   b_ptr  : [B, C_IN, HW]
#   out_ptr: [B, 2*C_IN, HW]
#     out[b, 2k,   hw] = a[b, k, hw]
#     out[b, 2k+1, hw] = b[b, k, hw]
#   Grid: (B * C_IN, ceil(HW / BLOCK))
# ---------------------------------------------------------------------------
@triton.jit
def interleave_full_kernel(
    a_ptr, b_ptr, out_ptr,
    B, C_IN, HW,
    BLOCK: tl.constexpr,
):
    pid_bc = tl.program_id(0)
    pid_hw = tl.program_id(1)

    b  = pid_bc // C_IN
    k  = pid_bc  % C_IN

    hw0     = pid_hw * BLOCK
    hw_offs = hw0 + tl.arange(0, BLOCK)
    mask    = hw_offs < HW

    src_off = b * (C_IN * HW) + k * HW + hw_offs
    a_vals  = tl.load(a_ptr + src_off, mask=mask, other=0.0)
    b_vals  = tl.load(b_ptr + src_off, mask=mask, other=0.0)

    out_base     = b * (2 * C_IN * HW)
    out_even_off = out_base + (2 * k)     * HW + hw_offs
    out_odd_off  = out_base + (2 * k + 1) * HW + hw_offs

    tl.store(out_ptr + out_even_off, a_vals, mask=mask)
    tl.store(out_ptr + out_odd_off,  b_vals, mask=mask)


# ---------------------------------------------------------------------------
# Replacement wrappers  (all @torch.fx.wrap, return a SINGLE tensor each)
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_sigmoid_mul_impl(in5, conv_out):
    """
    Fuses: tmp_3 = sigmoid(conv_out);  return in5 * tmp_3
    conv_out : [B, 40, 1, 1]
    in5      : [B, 40, 32, 24]
    Returns  : [B, 40, 32, 24]
    """
    out  = torch.empty_like(in5)
    B    = in5.shape[0]
    C    = in5.shape[1]
    HW   = in5.shape[2] * in5.shape[3]
    N    = B * C * HW
    BLK  = 1024
    fused_sigmoid_mul_kernel[((N + BLK - 1) // BLK,)](
        conv_out, in5, out, N, C, HW, BLOCK=BLK,
    )
    return out


@torch.fx.wrap
def fused_shuffle1_impl(in_2, in_4):
    """
    Fuses: cat([in_2, in_4], dim=1) + channel-shuffle → [B, 40, 64, 48]
    in_2, in_4 : [B, 20, 64, 48]
    Returns    : [B, 40, 64, 48]
    """
    B    = in_2.shape[0]
    C_IN = in_2.shape[1]   # 20
    HW   = in_2.shape[2] * in_2.shape[3]   # 64*48=3072
    out  = torch.empty((B, C_IN * 2, in_2.shape[2], in_2.shape[3]),
                       dtype=in_2.dtype, device=in_2.device)
    BLK  = 1024
    n_hw = (HW + BLK - 1) // BLK
    interleave_full_kernel[(B * C_IN, n_hw)](
        in_2, in_4, out, B, C_IN, HW, BLOCK=BLK,
    )
    return out


@torch.fx.wrap
def fused_shuffle2_impl(in_3, x):
    """
    Fuses: cat([in_3, x], dim=1) + channel-shuffle → [B, 80, 32, 24]
    in_3, x  : [B, 40, 32, 24]
    Returns  : [B, 80, 32, 24]
    """
    B    = in_3.shape[0]
    C_IN = in_3.shape[1]   # 40
    HW   = in_3.shape[2] * in_3.shape[3]   # 32*24=768
    out  = torch.empty((B, C_IN * 2, in_3.shape[2], in_3.shape[3]),
                       dtype=in_3.dtype, device=in_3.device)
    BLK  = 1024
    n_hw = (HW + BLK - 1) // BLK
    interleave_full_kernel[(B * C_IN, n_hw)](
        in_3, x, out, B, C_IN, HW, BLOCK=BLK,
    )
    return out