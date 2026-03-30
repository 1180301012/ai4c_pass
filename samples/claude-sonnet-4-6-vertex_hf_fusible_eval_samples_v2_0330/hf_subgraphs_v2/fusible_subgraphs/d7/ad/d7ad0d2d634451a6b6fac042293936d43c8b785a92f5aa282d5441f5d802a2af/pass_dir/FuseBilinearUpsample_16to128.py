"""
FuseBilinearUpsample_16to128.py

Replaces torch.nn.functional.interpolate(x, size=(128,128), mode='bilinear',
align_corners=False) with a Triton kernel for the fixed
16x16 -> 128x128 upsampling.
"""

import torch
import triton
import triton.language as tl


def pattern(x):
    return torch.nn.functional.interpolate(
        x, size=(128, 128), mode='bilinear', align_corners=False
    )


def replacement_args(x):
    return (x,)


@triton.jit
def bilinear_16to128_kernel(
    inp_ptr,
    out_ptr,
    N,
    H_IN:  tl.constexpr,
    W_IN:  tl.constexpr,
    H_OUT: tl.constexpr,
    W_OUT: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)

    h_start = pid_h * BLOCK_H
    h_range = h_start + tl.arange(0, BLOCK_H)
    w_range = tl.arange(0, W_OUT)
    h_mask  = h_range < H_OUT

    scale = H_IN / H_OUT

    h_in_f = -0.5 + (h_range.to(tl.float32) + 0.5) * scale
    h0_f   = tl.floor(h_in_f)
    h0     = h0_f.to(tl.int32)
    lh     = h_in_f - h0_f
    h0c    = tl.maximum(0, tl.minimum(H_IN - 1, h0))
    h1c    = tl.maximum(0, tl.minimum(H_IN - 1, h0 + 1))

    w_in_f = -0.5 + (w_range.to(tl.float32) + 0.5) * scale
    w0_f   = tl.floor(w_in_f)
    w0     = w0_f.to(tl.int32)
    lw     = w_in_f - w0_f
    w0c    = tl.maximum(0, tl.minimum(W_IN - 1, w0))
    w1c    = tl.maximum(0, tl.minimum(W_IN - 1, w0 + 1))

    base    = pid_n * H_IN * W_IN
    hmask2d = h_mask[:, None]

    v00 = tl.load(inp_ptr + base + h0c[:, None] * W_IN + w0c[None, :],
                  mask=hmask2d, other=0.0).to(tl.float32)
    v01 = tl.load(inp_ptr + base + h0c[:, None] * W_IN + w1c[None, :],
                  mask=hmask2d, other=0.0).to(tl.float32)
    v10 = tl.load(inp_ptr + base + h1c[:, None] * W_IN + w0c[None, :],
                  mask=hmask2d, other=0.0).to(tl.float32)
    v11 = tl.load(inp_ptr + base + h1c[:, None] * W_IN + w1c[None, :],
                  mask=hmask2d, other=0.0).to(tl.float32)

    lh2 = lh[:, None]
    lw2 = lw[None, :]
    result = (
        (1.0 - lh2) * (1.0 - lw2) * v00 +
        (1.0 - lh2) *        lw2  * v01 +
               lh2  * (1.0 - lw2) * v10 +
               lh2  *        lw2  * v11
    )

    out_idx = pid_n * H_OUT * W_OUT + h_range[:, None] * W_OUT + w_range[None, :]
    tl.store(out_ptr + out_idx, result, mask=hmask2d)


@torch.fx.wrap
def bilinear_upsample_16to128(x):
    B    = x.shape[0]
    C    = x.shape[1]
    H_in = x.shape[2]
    W_in = x.shape[3]
    N    = B * C
    H_out = 128
    W_out = 128
    BLOCK_H = 16

    out_f32 = torch.empty((N, H_out, W_out), dtype=torch.float32, device=x.device)
    x_flat  = x.contiguous().view(N, H_in, W_in)

    grid = (N, triton.cdiv(H_out, BLOCK_H))

    bilinear_16to128_kernel[grid](
        x_flat, out_f32, N,
        H_IN=H_in, W_IN=W_in, H_OUT=H_out, W_OUT=W_out,
        BLOCK_H=BLOCK_H,
    )

    out = out_f32.view(B, C, H_out, W_out)
    if x.dtype != torch.float32:
        out = out.to(x.dtype)
    return out


def replacement_func():
    return bilinear_upsample_16to128