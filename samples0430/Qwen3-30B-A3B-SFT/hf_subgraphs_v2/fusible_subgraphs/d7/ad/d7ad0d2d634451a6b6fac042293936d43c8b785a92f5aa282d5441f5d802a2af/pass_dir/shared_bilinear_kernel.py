"""
Shared Triton kernel for fused permute + reshape + bilinear interpolation.

Input:  y = linear_out  [B, H_IN*W_IN, C_IN]  (i.e., [B, 256, 512])
Output: z = [B, C_OUT, H_OUT, W_OUT] = [B, 48, 128, 128]

each output element z[b, c, h_out, w_out] = bilinear_interp(y[b, :, c*16*16..(c+1)*16*16], ...)

Grid: (B, C_OUT)  – each program handles one (batch, out_channel) pair
and loops over the full K = C_IN = 512 in BLOCK_K steps.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def bilinear_fused_kernel(
    src_ptr,          # input  [B, H_IN*W_IN, C_IN]
    weight_ptr,       # weight [C_OUT, C_IN]  (linear weight matrix)
    bias_ptr,         # bias   [C_OUT]
    out_ptr,          # output [B, C_OUT, H_OUT, W_OUT]
    H_IN:    tl.constexpr,   # 16
    W_IN:    tl.constexpr,   # 16
    H_OUT:   tl.constexpr,   # 128
    W_OUT:   tl.constexpr,   # 128
    C_IN:    tl.constexpr,   # 512
    C_OUT:   tl.constexpr,   # 768
    BLOCK_K: tl.constexpr,   # tile over the C_IN reduction dimension
):
    b     = tl.program_id(0)   # batch index
    c_out = tl.program_id(1)   # output channel  (0 .. C_OUT-1)

    # spatial scale factors (align_corners=False)
    scale_h = tl.cast(H_IN + 1, tl.float32) / tl.cast(H_OUT + 1, tl.float32)
    scale_w = tl.cast(W_IN + 1, tl.float32) / tl.cast(W_OUT + 1, tl.float32)

    # flat spatial offsets for every output pixel
    hw_flat = tl.arange(0, H_OUT * W_OUT)   # shape [H_OUT*W_OUT]
    h_out_i = hw_flat // W_OUT               # row index
    w_out_i = hw_flat % W_OUT                # col index

    # compute source (h, w) in the [H_IN, W_IN] source grid
    h_src = (h_out_i.to(tl.float32) + 0.5) * scale_h - 0.5
    w_src = (w_out_i.to(tl.float32) + 0.5) * scale_w - 0.5

    # floor + clamp to [0, H_IN-1] / [0, W_IN-1]
    h_floor = tl.floor(h_src).to(tl.int32)
    w_floor = tl.floor(w_src).to(tl.int32)
    h0 = tl.maximum(0, tl.minimum(H_IN - 1, h_floor))
    h1 = tl.maximum(0, tl.minimum(H_IN - 1, h_floor + 1))
    w0 = tl.maximum(0, tl.minimum(W_IN - 1, w_floor))
    w1 = tl.maximum(0, tl.minimum(W_IN - 1, w_floor + 1))

    # bilinear weights
    fh = h_src - h_floor.to(tl.float32)
    fw = w_src - w_floor.to(tl.float32)
    fh = tl.maximum(0.0, tl.minimum(1.0, fh))
    fw = tl.maximum(0.0, tl.minimum(1.0, fw))

    # absolute memory offsets for the 4 bilinear corners in the source tensor
    # src layout: [B, H_IN*W_IN, C_IN]  =>  flat = b*(H_IN*W_IN*C_IN) + n*C_IN + k
    n00 = h0 * W_IN + w0
    n01 = h0 * W_IN + w1
    n10 = h1 * W_IN + w0
    n11 = h1 * W_IN + w1

    out_offset = (b * C_OUT + c_out) * H_OUT * W_OUT

    # accumulate over the C_IN dimension in tiles of BLOCK_K
    acc = tl.zeros([H_OUT * W_OUT], dtype=tl.float32)
    for k_start in range(0, C_IN, BLOCK_K):
        k_off = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_off < C_IN

        # load source pixels for all 4 corners (shape [BLOCK_K])
        s00 = tl.load(src_ptr + b * (H_IN * W_IN * C_IN) + n00[:, None] * C_IN + k_off[None, :],
                      mask=k_mask[None, :], other=0.0)
        s01 = tl.load(src_ptr + b * (H_IN * W_IN * C_IN) + n01[:, None] * C_IN + k_off[None, :],
                      mask=k_mask[None, :], other=0.0)
        s10 = tl.load(src_ptr + b * (H_IN * W_IN * C_IN) + n10[:, None] * C_IN + k_off[None, :],
                      mask=k_mask[None, :], other=0.0)
        s11 = tl.load(src_ptr + b * (H_IN * W_IN * C_IN) + n11[:, None] * C_IN + k_off[None, :],
                      mask=k_mask[None, :], other=0.0)

        # load weight rows for c_out at all k positions
        # weight layout: [C_OUT, C_IN]  =>  flat = c_out*C_IN + k
        w00 = tl.load(weight_ptr + c_out * C_IN + k_off, mask=k_mask, other=0.0)
        w01 = tl.load(weight_ptr + c_out * C_IN + k_off, mask=k_mask, other=0.0)
        w10 = tl.load(weight_ptr + c_out * C_IN + k_off, mask=k_mask, other=0.0)
        w11 = tl.load(weight_ptr + c_out * C_IN + k_off, mask=k_mask, other=0.0)

        # accumulate (broadcast dot-product over spatial dimension)
        acc += tl.sum(s00 * w00[None, :], axis=1)
        acc += tl.sum(s01 * w01[None, :], axis=1)
        acc += tl.sum(s10 * w10[None, :], axis=1)
        acc += tl.sum(s11 * w11[None, :], axis=1)

    # add bias
    bias_val = tl.load(bias_ptr + c_out)
    acc += bias_val

    # store output (cast back to input dtype)
    tl.store(out_ptr + out_offset + hw_flat, acc, mask=None)


@torch.fx.wrap
def fused_bilinear_interp(y, weight, bias):
    """
    Fused permute(0,2,1) + reshape(B,-1,16,16) + bilinear-interpolate-to-128x128.

    Args:
        y      : linear output  [B, H_IN*W_IN, C_IN]  (e.g. [2, 256, 512])
        weight : linear weight   [C_OUT, C_IN]          (e.g. [768, 512])
        bias   : linear bias     [C_OUT]                (e.g. [768])
    Returns:
        out    : [B, C_OUT, H_OUT, W_OUT]               (e.g. [2, 48, 128, 128])
    """
    B     = y.shape[0]
    H_IN  = 16
    W_IN  = 16
    H_OUT = 128
    W_OUT = 128
    C_IN  = 512
    C_OUT = 768

    out = torch.empty((B, C_OUT, H_OUT, W_OUT), dtype=y.dtype, device=y.device)

    BLOCK_K = 64  # tile size over C_IN dimension

    bilinear_fused_kernel[(B, C_OUT)](
        y, weight, bias, out,
        H_IN=H_IN, W_IN=W_IN,
        H_OUT=H_OUT, W_OUT=W_OUT,
        C_IN=C_IN, C_OUT=C_OUT,
        BLOCK_K=BLOCK_K,
    )

    return out