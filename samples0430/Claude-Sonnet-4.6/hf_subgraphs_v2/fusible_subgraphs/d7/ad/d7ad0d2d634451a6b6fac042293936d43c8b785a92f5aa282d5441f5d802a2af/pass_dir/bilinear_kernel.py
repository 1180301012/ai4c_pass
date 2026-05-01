"""
Shared Triton kernel for fused permute + reshape + bilinear interpolation.

This fuses the sequence:
  linear_out [B, 256, C]
  -> .permute(0, 2, 1)          [B, C, 256]  (non-contiguous)
  -> .reshape(B, -1, 16, 16)    [B, C, 16, 16]  (may copy due to non-contiguous input)
  -> interpolate(size=(128,128), mode='bilinear', align_corners=False)  [B, C, 128, 128]

into a single Triton kernel that:
  - Reads directly from the contiguous linear_out [B, 256, C]
  - Treats spatial dim as 16x16 grid
  - Writes bilinear-upsampled result [B, C, 128, 128]

This avoids the reshape copy and reduces total memory traffic.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
    ],
    key=['B', 'C'],
)
@triton.jit
def bilinear_upsample_fused_kernel(
    x_ptr,    # Input: [B, 256, C] -- linear output, contiguous
    out_ptr,  # Output: [B, C, 128, 128]
    B, C,
    IN_H: tl.constexpr,   # = 16
    IN_W: tl.constexpr,   # = 16
    OUT_H: tl.constexpr,  # = 128
    OUT_W: tl.constexpr,  # = 128
):
    """
    One program per (b, c, oh) triple.
    Grid size: B * C * OUT_H
    Each program processes OUT_W output pixels (one full output row).
    """
    pid = tl.program_id(0)

    # Decompose pid: iterating oh fastest, then c, then b
    oh = pid % OUT_H
    bc = pid // OUT_H
    c = bc % C
    b = bc // C

    # ----- Height bilinear coordinates (scalars) -----
    scale_h = tl.cast(IN_H, tl.float32) / tl.cast(OUT_H, tl.float32)  # = 0.125
    ih_f = (tl.cast(oh, tl.float32) + 0.5) * scale_h - 0.5

    ih0_floor = tl.floor(ih_f)
    ih0 = ih0_floor.to(tl.int32)
    ih0 = tl.maximum(ih0, 0)
    ih1 = (ih0_floor.to(tl.int32)) + 1
    ih1 = tl.minimum(ih1, IN_H - 1)
    alpha = ih_f - ih0_floor  # weight for ih1 row; in [0, 1)

    # ----- Width bilinear coordinates (vectors of OUT_W=128) -----
    ow = tl.arange(0, OUT_W)  # [128]

    scale_w = tl.cast(IN_W, tl.float32) / tl.cast(OUT_W, tl.float32)  # = 0.125
    iw_f = (ow.to(tl.float32) + 0.5) * scale_w - 0.5  # [128]

    iw0_floor = tl.floor(iw_f)
    iw0 = iw0_floor.to(tl.int32)
    iw0 = tl.maximum(iw0, 0)
    iw1 = iw0_floor.to(tl.int32) + 1
    iw1 = tl.minimum(iw1, IN_W - 1)
    beta = iw_f - iw0_floor  # weight for iw1 col; in [0, 1)

    # ----- Load 4 input values per output pixel -----
    # x[b, s, c] where s = ih * IN_W + iw
    # Address: b * 256 * C + s * C + c
    x_base = b * (IN_H * IN_W) * C + c

    # ih0 is scalar, iw0/iw1 are vectors -> broadcast to [128]
    addr00 = x_base + (ih0 * IN_W + iw0) * C  # [128]
    addr01 = x_base + (ih0 * IN_W + iw1) * C  # [128]
    addr10 = x_base + (ih1 * IN_W + iw0) * C  # [128]
    addr11 = x_base + (ih1 * IN_W + iw1) * C  # [128]

    x00 = tl.load(x_ptr + addr00)  # [128], input dtype
    x01 = tl.load(x_ptr + addr01)
    x10 = tl.load(x_ptr + addr10)
    x11 = tl.load(x_ptr + addr11)

    # ----- Bilinear interpolation (compute in float32) -----
    x00f = x00.to(tl.float32)
    x01f = x01.to(tl.float32)
    x10f = x10.to(tl.float32)
    x11f = x11.to(tl.float32)

    # Bilinear formula:
    # out = (1-alpha)*((1-beta)*x00 + beta*x01) + alpha*((1-beta)*x10 + beta*x11)
    row0 = (1.0 - beta) * x00f + beta * x01f
    row1 = (1.0 - beta) * x10f + beta * x11f
    out_f = (1.0 - alpha) * row0 + alpha * row1

    # Cast back to input dtype before storing
    out_val = out_f.to(x00.dtype)

    # ----- Store output -----
    # out[b, c, oh, ow]: strides [C*128*128, 128*128, 128, 1]
    out_base = b * C * (OUT_H * OUT_W) + c * (OUT_H * OUT_W) + oh * OUT_W
    tl.store(out_ptr + out_base + ow, out_val)


@torch.fx.wrap
def fused_perm_reshape_bilinear(x):
    """
    x: [B, SEQ=256, C=768] -- output of torch.nn.functional.linear
    returns: [B, C, 128, 128] -- bilinear upsampled from [B, C, 16, 16]
    """
    B = x.shape[0]
    C = x.shape[2]
    OUT_H = 128
    OUT_W = 128

    out = torch.empty((B, C, OUT_H, OUT_W), dtype=x.dtype, device=x.device)

    # One program per (b, c, oh)
    grid = (B * C * OUT_H,)
    bilinear_upsample_fused_kernel[grid](
        x, out,
        B, C,
        IN_H=16,
        IN_W=16,
        OUT_H=OUT_H,
        OUT_W=OUT_W,
    )
    return out