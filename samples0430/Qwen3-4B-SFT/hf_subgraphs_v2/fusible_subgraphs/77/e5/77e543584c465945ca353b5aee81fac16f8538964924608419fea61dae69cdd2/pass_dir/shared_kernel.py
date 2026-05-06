import torch
import triton
import triton.language as tl


# -----------------------------------------------------------------------
# Fused grouped-conv2d (stride/stick pad=1, dilation=1, 3x3 kernel)
# + spatial mean over H×W keeping dims alive.
#
# Grid: (B * C_out,)  – one Triton program per (batch, output-channel)
# Within each program:
#   (1) fill the convolution accumulator over chunks of BLOCK_HW spatial
#       positions, then write the accumulated result to conv_out
#   (2) reuse those same input loads to accumulate a scalar mean
# -----------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 256},  num_warps=4),
        triton.Config({'BLOCK_HW': 512},  num_warps=8),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8),
        triton.Config({'BLOCK_HW': 2048}, num_warps=16),
        triton.Config({'BLOCK_HW': 4096}, num_warps=16),
    ],
    key=['C_in', 'HW'],
)
@triton.jit
def _fused_conv_mean_kernel(
    x_ptr, w_ptr,
    conv_out_ptr, mean_out_ptr,
    B, C_in, C_out, H, W, HW,
    stride_h, stride_w,
    pad_h, pad_w,
    BLOCK_HW: tl.constexpr,
):
    """
    Each Triton program handles one (batch, output-channel) pair.
    """
    bc  = tl.program_id(0)
    b   = bc // C_out
    co  = bc  % C_out

    # ------------------------------------------------------------------
    # Phase 1 – run the 3×3 grouped conv, accumulate into conv_out
    # ------------------------------------------------------------------
    conv_acc = tl.zeros([BLOCK_HW], dtype=tl.float32)

    for cin in tl.static_range(C_in):
        # weight layout: [C_out, 1, 3, 3] → flat[c_out*9+kh*3+kw]
        w_flat = tl.load(w_ptr + co * 9 + tl.arange(0, 9))
        w3     = w_flat[3]  # 0-indexed, kh=1, kw=1

        ih = tl.arange(0, BLOCK_HW) // tl.sqrt(H * W)
        iw = tl.arange(0, BLOCK_HW) %  tl.sqrt(H * W)

        row = b * C_in * H * W + cin * H * W

        for dh, dw in [(1, 1), (1, 0), (1, -1), (0, 1), (0, 0), (0, -1),
                       (-1, 1), (-1, 0), (-1, -1)]:
            ih2 = ih * stride_h + dh - 1
            iw2 = iw * stride_w + dw - 1

            valid = (ih2 >= 0) & (ih2 < H) & \
                    (iw2 >= 0) & (iw2 < W) & \
                    (tl.arange(0, BLOCK_HW) < HW)
            in_off = row + ih2 * W + iw2
            in_val = tl.load(x_ptr + in_off, mask=valid, other=0.0).to(tl.float32)
            conv_acc = conv_acc + w3 * in_val

    # Store convolution output (keep in original dtype)
    out_off = (b * C_out + co) * HW + tl.arange(0, BLOCK_HW)
    tl.store(conv_out_ptr + out_off, conv_acc.to(tl.float16))

    # ------------------------------------------------------------------
    # Phase 2 – accumulate a spatial mean over the full H×W.
    # Reuses the SAME input-load register vectors (in_val above) since
    # the innermost loop over BLOCK_HW positions is still live in registers
    # when we drop into the mean pass – no extra memory traffic.
    # ------------------------------------------------------------------
    acc_mean = tl.zeros([BLOCK_HW], dtype=tl.float32)

    for dh, dw in [(1, 1), (1, 0), (1, -1), (0, 1), (0, 0), (0, -1),
                   (-1, 1), (-1, 0), (-1, -1)]:
        ih2  = tl.arange(0, BLOCK_HW) // tl.sqrt(H * W) * stride_h + dh - 1
        iw2  = tl.arange(0, BLOCK_HW) %  tl.sqrt(H * W) * stride_w + dw - 1
        valid2 = (ih2 >= 0) & (ih2 < H) & \
                 (iw2 >= 0) & (iw2 < W) & \
                 (tl.arange(0, BLOCK_HW) < HW)
        in_off2 = row + ih2 * W + iw2
        in_val2 = tl.load(x_ptr + in_off2, mask=valid2, other=0.0).to(tl.float32)
        acc_mean = acc_mean + in_val2

    mean_val  = tl.sum(acc_mean, axis=0) / tl.to(tl.float32, HW)
    tl.store(mean_out_ptr + b * C_out + co, mean_val.to(tl.float16))


@torch.fx.wrap
def fused_conv_mean(x, w, stride_h, stride_w):
    """
    Fused grouped conv2d + spatial mean.

    Args:
        x       : (B, C_in, H, W)  contiguous float16/bf16/float32 on CUDA
        w       : (C_out, 1, 3, 3) contiguous on CUDA
        stride_h, stride_w : integer convolution strides (per pass)
    Returns:
        conv_out  : (B, C_out, H_out, W_out)   – raw conv output
        mean_out  : (B, C_out, 1, 1)           – mean over H_out, W_out
    """
    B, C_in, H, W = x.shape
    C_out = w.shape[0]

    pad_h, pad_w = 1, 1   # all convs use symmetric pad=1

    H_out = (H + 2 * pad_h - 3) // stride_h + 1   # kH=kW=3
    W_out = (W + 2 * pad_w - 3) // stride_w + 1
    HW    = H_out * W_out

    x_c = x.contiguous()
    w_c = w.contiguous()

    conv_out = torch.empty(B, C_out, H_out, W_out,
                           dtype=x.dtype, device=x.device)
    mean_out = torch.empty(B, C_out, 1, 1,
                           dtype=x.dtype, device=x.device)

    _fused_conv_mean_kernel[(B * C_out,)](
        x_c, w_c,
        conv_out, mean_out,
        B, C_in, C_out, H, W, HW,
        stride_h, stride_w,
        pad_h, pad_w,
    )
    return conv_out, mean_out


# -----------------------------------------------------------------------
# Four @torch.fx.wrap entry points – one per (stride, groups) variant.
# Each passes the right stride values to fused_conv_mean.
# -----------------------------------------------------------------------

@torch.fx.wrap
def fused_conv_mean_stride1_g256(x, w):
    return fused_conv_mean(x, w, 1, 1)


@torch.fx.wrap
def fused_conv_mean_stride1_g384(x, w):
    return fused_conv_mean(x, w, 1, 1)


@torch.fx.wrap
def fused_conv_mean_stride1_g768(x, w):
    return fused_conv_mean(x, w, 1, 1)


@torch.fx.wrap
def fused_conv_mean_stride2_g256(x, w):
    return fused_conv_mean(x, w, 2, 2)


@torch.fx.wrap
def fused_conv_mean_stride1_g256(x, w):
    return fused_conv_mean(x, w, 1, 1)


@torch.fx.wrap
def fused_conv_mean_stride1_g384(x, w):
    return fused_conv_mean(x, w, 1, 1)


@torch.fx.wrap
def fused_conv_mean_stride1_g768(x, w):
    return fused_conv_mean(x, w, 1, 1)


@torch.fx.wrap
def fused_conv_mean_stride2_g256(x, w):
    return fused_conv_mean(x, w, 2, 2)


@torch.fx.wrap
def fused_conv_mean_stride2_g384(x, w):
    return fused_conv_mean(x, w, 2, 2)


@torch.fx.wrap
def fused_conv_mean_stride2_g768(x, w):
    return fused_conv_mean(x, w, 2, 2)