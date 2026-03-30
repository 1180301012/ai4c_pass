"""
Shared Triton kernel for fused depthwise conv2d (3x3, pad=1) + spatial mean.

Pattern:
    out = conv2d(x, w, bias=None, stride, padding=(1,1), dilation=(1,1), groups=C)
    mean = out.mean((2, 3), keepdim=True)
    return out, mean

Fusion benefit: avoids a second global-memory read of `out` when computing mean.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64},   num_warps=2),
        triton.Config({'BLOCK_SIZE': 128},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['C', 'H_in', 'W_in'],
)
@triton.jit
def dw_conv3x3_mean_kernel(
    X, W, Out, SumBuf,
    B, C, H_in, W_in, H_out, W_out,
    SH: tl.constexpr, SW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Each Triton program handles one (batch, channel) pair and a spatial tile.

    Grid: (B*C, ceil(H_out*W_out / BLOCK_SIZE))

    Accumulates partial spatial sums into SumBuf[bc] via atomic_add.
    After the kernel, the caller divides SumBuf by (H_out*W_out) to get the mean.
    """
    bc   = tl.program_id(0)
    tile = tl.program_id(1)

    b = bc // C
    c = bc % C

    offs = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    hw   = H_out * W_out
    mask = offs < hw

    # Flatten spatial index → (h, w)
    h_pos = offs // W_out
    w_pos = offs % W_out

    x_base   = b * C * H_in * W_in   + c * H_in * W_in
    out_base = b * C * H_out * W_out + c * H_out * W_out

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # 3×3 depthwise convolution (unrolled at compile time)
    for kh in range(3):
        ih   = h_pos * SH + kh - 1          # input row (may be < 0)
        h_ok = (ih >= 0) & (ih < H_in)

        for kw in range(3):
            iw   = w_pos * SW + kw - 1      # input col (may be < 0)
            w_ok = (iw >= 0) & (iw < W_in)

            wgt = tl.load(W + c * 9 + kh * 3 + kw).to(tl.float32)
            xv  = tl.load(X + x_base + ih * W_in + iw,
                          mask=mask & h_ok & w_ok,
                          other=0.0).to(tl.float32)
            acc = acc + xv * wgt

    # Store conv output (auto-cast from float32 → pointee dtype)
    tl.store(Out + out_base + offs, acc, mask=mask)

    # Accumulate partial sum for mean (float32 atomic)
    ps = tl.sum(tl.where(mask, acc, tl.zeros([BLOCK_SIZE], dtype=tl.float32)))
    tl.atomic_add(SumBuf + bc, ps)


# ---------------------------------------------------------------------------
# Python wrapper (shared by all pass files)
# ---------------------------------------------------------------------------

def fused_dw_conv_mean(in_0, in_1, stride_h: int, stride_w: int):
    """
    in_0: weight  [C, 1, 3, 3]  (may be on CPU)
    in_1: input   [B, C, H, W]  (on CUDA)

    Returns (conv_out, mean_out) matching torch.conv2d + .mean((2,3), keepdim=True).
    """
    x = in_1
    w = in_0.to(x.device, x.dtype)           # move weight to same device/dtype

    B, C, H_in, W_in = x.shape
    # With pad=1, kernel=3: H_out = floor((H_in + 2*1 - 3) / stride) + 1
    H_out = (H_in + 2 - 3) // stride_h + 1
    W_out = (W_in + 2 - 3) // stride_w + 1

    out     = torch.empty((B, C, H_out, W_out), dtype=x.dtype, device=x.device)
    sum_buf = torch.zeros(B * C, dtype=torch.float32, device=x.device)

    grid = lambda meta: (B * C, triton.cdiv(H_out * W_out, meta['BLOCK_SIZE']))

    dw_conv3x3_mean_kernel[grid](
        x, w, out, sum_buf,
        B, C, H_in, W_in, H_out, W_out,
        stride_h, stride_w,
    )

    mean_out = (sum_buf / (H_out * W_out)).to(x.dtype).view(B, C, 1, 1)
    return out, mean_out