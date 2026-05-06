"""
Fused pass: 1x1 depthwise conv + add + add + BatchNorm (inference) + spatial mean

The computation is:
  conv_out   = X * W + B          (1x1 depthwise = channel-wise scale+bias)
  tmp7       = Y + conv_out
  tmp8       = tmp7 + in7         (or tmp8 = tmp7 + inX)
  tmp9       = (tmp8 - mean) / sqrt(var + eps) * weight + bias  (BN inference)
  tmp10      = tmp9.mean((2,3), keepdim=True)

All ops are element-wise except the final spatial reduction.
A single Triton kernel eliminates all intermediate tensor allocations.
"""

import torch
import triton
import triton.language as tl


def pattern(X, W, B, Y, in7, inX, bn_mean, bn_var, bn_bias, bn_weight):
    """
    Matches the RepVit-token-mixer post-conv block.
    Argument names are chosen to match the FX graph topology.
    """
    conv_out = torch.conv2d(X, W, B, (1, 1), (0, 0), (1, 1), groups=groups)
    tmp7 = Y + conv_out
    tmp8 = tmp7 + in7
    tmp9 = torch.nn.functional.batch_norm(tmp8, bn_mean, bn_var, bn_weight, bn_bias, False, 0.1, 1e-05)
    tmp10 = tmp9.mean((2, 3), keepdim=True)
    return tmp9, tmp10


def replacement_args(X, W, B, Y, in7, inX, bn_mean, bn_var, bn_bias, bn_weight):
    return (X, W, B, Y, in7, inX, bn_mean, bn_var, bn_bias, bn_weight)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64},  num_warps=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=4),
        triton.Config({'BLOCK_HW': 256}, num_warps=4),
        triton.Config({'BLOCK_HW': 512}, num_warps=4),
        triton.Config({'BLOCK_HW': 1024}, num_warps=4),
        triton.Config({'BLOCK_HW': 2048}, num_warps=8),
        triton.Config({'BLOCK_HW': 4096}, num_warps=8),
        triton.Config({'BLOCK_HW': 8192}, num_warps=8),
    ],
    key=['HW'],
)
@triton.jit
def _fused_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    in7_ptr, inX_ptr,
    bn_mean_ptr, bn_var_ptr, bn_bias_ptr, bn_weight_ptr,
    out9_ptr, out10_ptr,
    B, C, HW,
    BLOCK_HW: tl.constexpr,
):
    """
    One program per (batch, channel) pair.
    Loads W, B, bn_mean, bn_var, bn_bias, bn_weight once per (b, c),
    then processes all HW spatial elements in BLOCK_HW chunks.
    """
    pid = tl.program_id(0)
    b = pid // C
    c = pid % C

    # Load per-channel scalars (shared across all spatial elements)
    w      = tl.load(W_ptr      + c).to(tl.float32)
    bc     = tl.load(bn_mean_ptr + c).to(tl.float32)
    bv     = tl.load(bn_var_ptr  + c).to(tl.float32)
    bb     = tl.load(bn_bias_ptr + c).to(tl.float32)
    bw     = tl.load(bn_weight_ptr + c).to(tl.float32)

    batch_base = pid * HW

    # Accumulator for the spatial mean over this (b, c) slice
    sum_batch = tl.zeros([BLOCK_HW], dtype=tl.float32)

    # Process spatial elements in BLOCK_HW-sized tiles
    for hw_start in range(0, HW, BLOCK_HW):
        offs = hw_start + tl.arange(0, BLOCK_HW)
        mask = offs < HW

        # Masked loads: out-of-bounds elements read as 0 (don't affect BN/sum)
        X = tl.load(X_ptr   + batch_base + offs, mask=mask, other=0.0).to(tl.float32)
        W = tl.load(W_ptr   + c).to(tl.float32)          # scalar broadcast
        B = tl.load(B_ptr   + c).to(tl.float32)          # scalar broadcast
        Y = tl.load(Y_ptr   + batch_base + offs, mask=mask, other=0.0).to(tl.float32)
        i7 = tl.load(in7_ptr + batch_base + offs, mask=mask, other=0.0).to(tl.float32)
        Xi = tl.load(inX_ptr + batch_base + offs, mask=mask, other=0.0).to(tl.float32)

        # Fused: 1x1 depthwise conv, two adds, BN
        x_vals = (X * w + B) + Y + i7
        # BN: (x - mean) / sqrt(var + eps) * weight + bias
        x_norm = (x_vals - bc) / tl.sqrt(bv + 1e-5) * bw + bb

        tl.store(out9_ptr + batch_base + offs, x_norm)

        # Spatial sum (masked elements contribute 0)
        sum_batch = sum_batch + tl.where(mask, x_norm, tl.zeros([BLOCK_HW], dtype=tl.float32))

    # Write spatial mean
    mean_val = tl.sum(sum_batch, axis=0) / HW
    tl.store(out10_ptr + pid, mean_val)


@torch.fx.wrap
def fused_conv_add_bn_mean(X, W, B, Y, in7, inX, bn_mean, bn_var, bn_bias, bn_weight):
    """
    Replaces: 1x1 depthwise conv → add → add → BN (inference) → spatial mean.
    Returns:  (normalized_output [B,C,H,W], spatial_mean [B,C,1,1])
    """
    B, C, H, W = X.shape
    HW = H * W

    device = X.device
    dtype  = X.dtype

    out9  = torch.empty_like(X)
    out10 = torch.empty((B, C, 1, 1), dtype=dtype, device=device)

    grid = (B * C,)

    _fused_kernel[grid](
        X, W, B, Y,
        in7, inX,
        bn_mean, bn_var, bn_bias, bn_weight,
        out9, out10,
        B, C, HW,
    )

    return out9, out10


def replacement_func():
    return fused_conv_add_bn_mean