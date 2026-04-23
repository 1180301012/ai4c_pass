import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_W': 32, 'BLOCK_H': 1}, num_warps=4),
        triton.Config({'BLOCK_W': 64, 'BLOCK_H': 1}, num_warps=4),
        triton.Config({'BLOCK_W': 128, 'BLOCK_H': 1}, num_warps=8),
        triton.Config({'BLOCK_W': 256, 'BLOCK_H': 1}, num_warps=8),
    ],
    key=['W_out'],
)
@triton.jit
def depthwise_conv3x3_mean_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    mean_ptr,
    N,
    C,
    H,
    W,
    H_out,
    W_out,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    x_stride_n,
    x_stride_c,
    x_stride_h,
    x_stride_w,
    w_stride_c,
    w_stride_kh,
    w_stride_kw,
    out_stride_n,
    out_stride_c,
    out_stride_h,
    out_stride_w,
    mean_stride_n,
    mean_stride_c,
    BLOCK_W: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    n = pid1 // C
    c = pid1 % C

    ow_block = pid0 * BLOCK_W + tl.arange(0, BLOCK_W)
    oh_block = tl.arange(0, BLOCK_H)

    k0 = tl.load(w_ptr + c * w_stride_c + 0 * w_stride_kh + 0 * w_stride_kw)
    k1 = tl.load(w_ptr + c * w_stride_c + 0 * w_stride_kh + 1 * w_stride_kw)
    k2 = tl.load(w_ptr + c * w_stride_c + 0 * w_stride_kh + 2 * w_stride_kw)
    k3 = tl.load(w_ptr + c * w_stride_c + 1 * w_stride_kh + 0 * w_stride_kw)
    k4 = tl.load(w_ptr + c * w_stride_c + 1 * w_stride_kh + 1 * w_stride_kw)
    k5 = tl.load(w_ptr + c * w_stride_c + 1 * w_stride_kh + 2 * w_stride_kw)
    k6 = tl.load(w_ptr + c * w_stride_c + 2 * w_stride_kh + 0 * w_stride_kw)
    k7 = tl.load(w_ptr + c * w_stride_c + 2 * w_stride_kh + 1 * w_stride_kw)
    k8 = tl.load(w_ptr + c * w_stride_c + 2 * w_stride_kh + 2 * w_stride_kw)

    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    mean_acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

    oh = pid0 * 0  # keep scalar-like variable available for compilation
    for oh in range(0, H_out):
        ih_base = oh * stride_h - pad_h
        ih0 = ih_base + 0
        ih1 = ih_base + 1
        ih2 = ih_base + 2

        iw_base = ow_block * stride_w - pad_w
        iw0 = iw_base + 0
        iw1 = iw_base + 1
        iw2 = iw_base + 2

        x_base = x_ptr + n * x_stride_n + c * x_stride_c

        mask_oh = oh < H_out
        mask0_h = (ih0 >= 0) & (ih0 < H)
        mask1_h = (ih1 >= 0) & (ih1 < H)
        mask2_h = (ih2 >= 0) & (ih2 < H)
        mask0_w = (iw0 >= 0) & (iw0 < W)
        mask1_w = (iw1 >= 0) & (iw1 < W)
        mask2_w = (iw2 >= 0) & (iw2 < W)
        out_mask = mask_oh & (ow_block < W_out)

        row0_col0 = tl.load(x_base + ih0 * x_stride_h + iw0 * x_stride_w, mask=mask0_h & mask0_w & out_mask, other=0.0)
        row0_col1 = tl.load(x_base + ih0 * x_stride_h + iw1 * x_stride_w, mask=mask0_h & mask1_w & out_mask, other=0.0)
        row0_col2 = tl.load(x_base + ih0 * x_stride_h + iw2 * x_stride_w, mask=mask0_h & mask2_w & out_mask, other=0.0)
        row1_col0 = tl.load(x_base + ih1 * x_stride_h + iw0 * x_stride_w, mask=mask1_h & mask0_w & out_mask, other=0.0)
        row1_col1 = tl.load(x_base + ih1 * x_stride_h + iw1 * x_stride_w, mask=mask1_h & mask1_w & out_mask, other=0.0)
        row1_col2 = tl.load(x_base + ih1 * x_stride_h + iw2 * x_stride_w, mask=mask1_h & mask2_w & out_mask, other=0.0)
        row2_col0 = tl.load(x_base + ih2 * x_stride_h + iw0 * x_stride_w, mask=mask2_h & mask0_w & out_mask, other=0.0)
        row2_col1 = tl.load(x_base + ih2 * x_stride_h + iw1 * x_stride_w, mask=mask2_h & mask1_w & out_mask, other=0.0)
        row2_col2 = tl.load(x_base + ih2 * x_stride_h + iw2 * x_stride_w, mask=mask2_h & mask2_w & out_mask, other=0.0)

        vals = (
            row0_col0.to(tl.float32) * k0.to(tl.float32) +
            row0_col1.to(tl.float32) * k1.to(tl.float32) +
            row0_col2.to(tl.float32) * k2.to(tl.float32) +
            row1_col0.to(tl.float32) * k3.to(tl.float32) +
            row1_col1.to(tl.float32) * k4.to(tl.float32) +
            row1_col2.to(tl.float32) * k5.to(tl.float32) +
            row2_col0.to(tl.float32) * k6.to(tl.float32) +
            row2_col1.to(tl.float32) * k7.to(tl.float32) +
            row2_col2.to(tl.float32) * k8.to(tl.float32)
        )

        out_offs = out_ptr + n * out_stride_n + c * out_stride_c + oh * out_stride_h + ow_block * out_stride_w
        tl.store(out_offs, vals, mask=out_mask)
        mean_acc += tl.sum(tl.where(out_mask, vals, 0.0), axis=0)

    denom = H_out * W_out
    mean_val = tl.sum(mean_acc, axis=0) / denom
    tl.store(mean_ptr + n * mean_stride_n + c * mean_stride_c, mean_val)


@torch.fx.wrap
def fused_depthwise_conv_mean(x, w, stride_h, stride_w, groups):
    if groups != x.shape[1] or w.shape[1] != 1 or w.shape[2] != 3 or w.shape[3] != 3:
        raise NotImplementedError('Only depthwise 3x3 conv with groups==channels is supported')

    N, C, H, W = x.shape
    kH = 3
    kW = 3
    pad_h = 1
    pad_w = 1
    dil_h = 1
    dil_w = 1
    H_out = (H + 2 * pad_h - dil_h * (kH - 1) - 1) // stride_h + 1
    W_out = (W + 2 * pad_w - dil_w * (kW - 1) - 1) // stride_w + 1

    out = torch.empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)
    mean = torch.empty((N, C, 1, 1), device=x.device, dtype=torch.float32)

    grid = lambda META: (triton.cdiv(W_out, META['BLOCK_W']), N * C)

    depthwise_conv3x3_mean_kernel[grid](
        x,
        w,
        out,
        mean,
        N,
        C,
        H,
        W,
        H_out,
        W_out,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        w.stride(0),
        w.stride(2),
        w.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        mean.stride(0),
        mean.stride(1),
    )

    if x.dtype == torch.float16:
        mean_out = mean.to(torch.float16)
    elif x.dtype == torch.bfloat16:
        mean_out = mean.to(torch.bfloat16)
    else:
        mean_out = mean
    return out, mean_out


def shared_replacement_func():
    return fused_depthwise_conv_mean